#!/usr/bin/env python3
import argparse
import os
import re
from typing import List, Tuple, Dict, Any, Optional

import requests

# ======================================================================
# ПЕРЕМЕННЫЕ СРЕДЫ
# ======================================================================
# Можно не передавать флаги, а просто задать:
#   ABS_BASE_URL=http://localhost:13378
#   ABS_TOKEN=ВАШ_ТОКЕН
#   ABS_DRY_RUN=1  (или TRUE/true/yes)
#
# Пример запуска:
#   export ABS_BASE_URL="http://localhost:13378"
#   export ABS_TOKEN="ВАШ_ТОКЕН"
#   export ABS_DRY_RUN="1"
#   python fix_abs_metadata.py
# ======================================================================

ABS_BASE_URL_ENV = os.environ.get("ABS_BASE_URL", "http://192.168.1.161:16378")
ABS_TOKEN_ENV = os.environ.get("ABS_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlJZCI6IjY1NThhZDE5LWM2MDUtNDE3Ni1iMjY2LTE3Y2QzNzE0NjE1MCIsIm5hbWUiOiI2NjY2IiwidHlwZSI6ImFwaSIsImlhdCI6MTc2NDc4ODMyOH0.3xd1NmYZXPvmrUA4CF5Eym0RsUg2VzzplBXDQcxvGuQ")

ABS_DRY_RUN_ENV = os.environ.get("ABS_DRY_RUN", "0")
DEFAULT_DRY_RUN = ABS_DRY_RUN_ENV.lower() in ("1", "true", "yes", "y", "on")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Исправление метаданных книг в Audiobookshelf "
                    "на основе тегов аудиофайлов"
    )
    parser.add_argument(
        "--base-url",
        default=ABS_BASE_URL_ENV,
        help="Базовый URL Audiobookshelf "
             f"(по умолчанию: {ABS_BASE_URL_ENV!r} "
             "или переменная окружения ABS_BASE_URL)",
    )
    parser.add_argument(
        "--token",
        default=ABS_TOKEN_ENV,
        help="Bearer-токен Audiobookshelf "
             "(по умолчанию из переменной ABS_TOKEN)",
    )
    parser.add_argument(
        "--library-id",
        help="Обрабатывать только одну библиотеку (ID). "
             "По умолчанию — все библиотеки с mediaType=book",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Размер батча для batch/get (по умолчанию: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DEFAULT_DRY_RUN,
        help=("Если указан — только показывать изменения (без PATCH). "
              f"По умолчанию: {'включён' if DEFAULT_DRY_RUN else 'выключен'}, "
              "значение можно задать через ABS_DRY_RUN=0/1/true/false"),
    )
    return parser.parse_args()


def chunked(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_session(base_url: str, token: Optional[str]) -> Tuple[requests.Session, str]:
    base_url = base_url.rstrip("/")
    session = requests.Session()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session, base_url


# --- Разбор строки альбома -------------------------------------------------


def _split_authors(author_part: str) -> List[str]:
    """
    Делим строку авторов по запятым:
        'Нил Гейман, Терри Пратчетт' → ['Нил Гейман', 'Терри Пратчетт']
    Если запятых нет — возвращаем один элемент.
    """
    author_part = author_part.strip()
    if not author_part:
        return []
    parts = re.split(r"\s*,\s*", author_part)
    return [p for p in (s.strip() for s in parts) if p]


def extract_authors_title_from_album(album: str) -> Tuple[List[str], Optional[str]]:
    """
    Пытаемся разобрать строку вида:
        'Терри Пратчетт. «Патриот»'
        'Нил Гейман, Терри Пратчетт. «Добрые предзнаменования»'
        'Терри Пратчетт: "Патриот"'
        'Терри Пратчетт - Патриот'
    Возвращаем (authors_list, title). authors_list может быть пустым.
    """
    if not album:
        return [], None

    album = album.strip()

    # 1) Вариант с русскими кавычками «Название»
    open_q = album.find("«")
    close_q = album.rfind("»")
    if open_q != -1 and close_q != -1 and close_q > open_q + 1:
        author_part = album[:open_q].strip()
        title = album[open_q + 1:close_q].strip()
        author_part = re.sub(r"[.:,\-–—]+$", "", author_part).strip()
        authors = _split_authors(author_part)
        return authors, (title or None)

    # 2) Вариант с обычными кавычками "Название"
    open_q = album.find('"')
    close_q = album.rfind('"')
    if open_q != -1 and close_q != -1 and close_q > open_q + 1:
        author_part = album[:open_q].strip()
        title = album[open_q + 1:close_q].strip()
        author_part = re.sub(r"[.:,\-–—]+$", "", author_part).strip()
        authors = _split_authors(author_part)
        return authors, (title or None)

    # 3) Fallback: 'Автор - Название'
    if " - " in album:
        author_part, title = album.split(" - ", 1)
        authors = _split_authors(author_part)
        return authors, (title.strip() or None)

    # 4) Fallback: 'Автор — Название' (длинное тире)
    for dash in (" — ", " – "):
        if dash in album:
            author_part, title = album.split(dash, 1)
            authors = _split_authors(author_part)
            return authors, (title.strip() or None)

    return [], None


# --- Работа с API ----------------------------------------------------------


def get_book_libraries(
    session: requests.Session,
    base_url: str,
    only_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Получаем все библиотеки с mediaType='book' либо одну по ID."""
    resp = session.get(f"{base_url}/api/libraries")
    resp.raise_for_status()
    data = resp.json()
    libs = data.get("libraries", [])

    result = []
    for lib in libs:
        if only_id and lib["id"] != only_id:
            continue
        if lib.get("mediaType") == "book":
            result.append(lib)
    return result


def get_library_item_ids(
    session: requests.Session,
    base_url: str,
    library_id: str,
) -> List[str]:
    """
    Забираем ВСЕ элементы библиотеки.
    limit=0 → без лимита, все результаты сразу.
    """
    params = {"limit": 0}
    resp = session.get(f"{base_url}/api/libraries/{library_id}/items", params=params)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    return [item["id"] for item in results if item.get("mediaType") == "book"]


def batch_get_items(
    session: requests.Session,
    base_url: str,
    item_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    POST /api/items/batch/get – получаем расширенную информацию,
    включая audioFiles[].metaTags.
    """
    if not item_ids:
        return []
    resp = session.post(
        f"{base_url}/api/items/batch/get",
        json={"libraryItemIds": item_ids},
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("libraryItems", [])


def patch_book_metadata(
    session: requests.Session,
    base_url: str,
    item_id: str,
    metadata_updates: Dict[str, Any],
    dry_run: bool = True,
) -> None:
    """
    PATCH /api/items/<ID>/media c полем metadata.
    """
    payload = {"metadata": metadata_updates}
    if dry_run:
        # Ничего не отправляем — только логика просмотра.
        return

    resp = session.patch(f"{base_url}/api/items/{item_id}/media", json=payload)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"  !!! Ошибка PATCH /items/{item_id}/media: {e} — {resp.text}")


# --- Основная логика -------------------------------------------------------


def process_book_item(
    item: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    На вход — libraryItem (из batch/get).
    Возвращаем:
      - dict с изменённым metadata (только изменяемые поля) или None, если ничего не нужно менять
      - dict с информацией для лога
    """
    media = item.get("media") or {}
    meta = media.get("metadata") or {}
    audio_files = media.get("audioFiles") or []

    log_info: Dict[str, Any] = {
        "item_id": item.get("id"),
        "path": item.get("path"),
        "old_title": meta.get("title"),
        "old_author": None,
        "old_narrator": None,
        "album": None,
        "artist_tag": None,
    }

    authors_meta = meta.get("authors") or []
    if authors_meta:
        names = [a.get("name") for a in authors_meta if a.get("name")]
        log_info["old_author"] = ", ".join(names) if names else None

    narrators = meta.get("narrators") or []
    if narrators:
        log_info["old_narrator"] = narrators[0]

    # Ищем первый аудиофайл, у которого есть metaTags.tagAlbum
    meta_tags = None
    for af in audio_files:
        tags = af.get("metaTags") or {}
        if tags.get("tagAlbum"):
            meta_tags = tags
            break

    if not meta_tags:
        return None, log_info

    tag_album = meta_tags.get("tagAlbum") or ""
    tag_artist = meta_tags.get("tagArtist") or ""
    log_info["album"] = tag_album
    log_info["artist_tag"] = tag_artist

    authors_from_album, title_from_album = extract_authors_title_from_album(tag_album)

    # Если не смогли распарсить (нет ни авторов, ни названия) — ничего не меняем
    if not authors_from_album or not title_from_album:
        return None, log_info

    new_title = title_from_album
    new_authors = authors_from_album
    new_narrator = tag_artist.strip() if tag_artist else None

    updates: Dict[str, Any] = {}

    # Title
    if new_title and new_title != meta.get("title"):
        updates["title"] = new_title

    # Authors (список авторов)
    new_authors_objs = [{"name": name} for name in new_authors]
    # Можно просто перезаписать авторов — мы специально не пытаемся сохранить старые ID/slug
    if new_authors_objs:
        updates["authors"] = new_authors_objs

    # Narrators
    curr_narrator_name = narrators[0] if narrators else None
    if new_narrator and new_narrator != curr_narrator_name:
        updates["narrators"] = [new_narrator]

    return (updates or None), log_info


def main() -> None:
    args = parse_args()
    session, base_url = build_session(args.base_url, args.token)

    print("Используем настройки:")
    print(f"  base_url  = {base_url!r}")
    print(f"  token     = {'<указан>' if args.token else '<НЕ указан>'}")
    print(f"  dry_run   = {args.dry_run}")
    print()

    if not args.token:
        print("ВНИМАНИЕ: токен не указан (--token или ABS_TOKEN). "
              "Если сервер требует авторизацию, запросы будут падать.\n")

    # 1. Получаем библиотеки
    libraries = get_book_libraries(session, base_url, args.library_id)
    if not libraries:
        print("Книжные библиотеки не найдены (mediaType=book).")
        return

    print(f"Найдено библиотек (book): {len(libraries)}")
    if args.dry_run:
        print("Режим: DRY-RUN — изменения НЕ отправляются.\n")
    else:
        print("Режим: реальные изменения (PATCH /items/<ID>/media).\n")

    for lib in libraries:
        lib_id = lib["id"]
        lib_name = lib.get("name")
        print(f"=== Библиотека {lib_name!r} ({lib_id}) ===")

        item_ids = get_library_item_ids(session, base_url, lib_id)
        print(f"  Книг в библиотеке: {len(item_ids)}")

        total_updates = 0
        total_touched = 0

        for batch in chunked(item_ids, args.batch_size):
            items = batch_get_items(session, base_url, batch)
            for item in items:
                if item.get("mediaType") != "book":
                    continue

                updates, info = process_book_item(item)
                if updates is None:
                    continue

                total_touched += 1

                print(f"\nКнига: {info.get('path')}")
                print(f"  ID: {info.get('item_id')}")
                print(f"  Album (из файла): {info.get('album')}")
                print(f"  Artist (из файла, чтец?): {info.get('artist_tag')}")
                print(f"  Старый title: {info.get('old_title')!r}")
                print(f"  Старый author: {info.get('old_author')!r}")
                print(f"  Старый narrator: {info.get('old_narrator')!r}")

                new_title = updates.get("title", info.get("old_title"))
                if "authors" in updates and updates["authors"]:
                    new_author_str = ", ".join(a["name"] for a in updates["authors"])
                else:
                    new_author_str = info.get("old_author")

                if "narrators" in updates and updates["narrators"]:
                    new_narrator = updates["narrators"][0]
                else:
                    new_narrator = info.get("old_narrator")

                print(f"  Новый title: {new_title!r}")
                print(f"  Новый author: {new_author_str!r}")
                print(f"  Новый narrator: {new_narrator!r}")

                patch_book_metadata(
                    session,
                    base_url,
                    item_id=info["item_id"],
                    metadata_updates=updates,
                    dry_run=args.dry_run,
                )
                total_updates += 1

        print(f"\nИтого по библиотеке {lib_name!r}:")
        print(f"  Книг с найденным 'Album': {total_touched}")
        print(f"  Книг, для которых отправлен PATCH (или был бы в dry-run): {total_updates}")
        print("")


if __name__ == "__main__":
    main()
