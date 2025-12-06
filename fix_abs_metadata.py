#!/usr/bin/env python3
import argparse
import os
import re
import json
import tempfile
from typing import List, Tuple, Dict, Any, Optional
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    Image = None

import requests

# ======================================================================
# ПЕРЕМЕННЫЕ СРЕДЫ
# ======================================================================
# Можно не передавать флаги, а просто задать:
#   ABS_BASE_URL=http://localhost:13378
#   ABS_TOKEN=ВАШ_ТОКЕН
#   ABS_DRY_RUN=1  (или  TRUE/true/yes)
#   ABS_USE_FANTLAB=1  (включить FantLab по умолчанию)
#
# КЕШ СОСТОЯНИЯ:
#   ABS_CACHE_FILE=abs_metadata_state.json
#   ABS_DISABLE_CACHE=1
#
# Пример запуска:
#   export ABS_BASE_URL="http://localhost:13378"
#   export ABS_TOKEN="ВАШ_ТОКЕН"
#   export ABS_DRY_RUN="1"
#   export ABS_USE_FANTLAB="1"
#   export ABS_CACHE_FILE="abs_metadata_state.json"
#   python fix_abs_metadata.py
# ======================================================================

ABS_BASE_URL_ENV = os.environ.get("ABS_BASE_URL", "http://192.168.1.161:16378")
ABS_TOKEN_ENV = os.environ.get(
    "ABS_TOKEN",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlJZCI6IjY1NThhZDE5LWM2MDUtNDE3Ni1iMjY2LTE3Y2QzNzE0NjE1MCIsIm5hbWUiOiI2NjY2IiwidHlwZSI6ImFwaSIsImlhdCI6MTc2NDc4ODMyOH0.3xd1NmYZXPvmrUA4CF5Eym0RsUg2VzzplBXDQcxvGuQ"
)

ABS_DRY_RUN_ENV = os.environ.get("ABS_DRY_RUN", "0")
DEFAULT_DRY_RUN = ABS_DRY_RUN_ENV.lower() in ("1", "true", "yes", "y", "on")

ABS_USE_FANTLAB_ENV = os.environ.get("ABS_USE_FANTLAB", "1")
DEFAULT_USE_FANTLAB = ABS_USE_FANTLAB_ENV.lower() in ("1", "true", "yes", "y", "on")

ABS_CACHE_FILE_ENV = os.environ.get("ABS_CACHE_FILE", "abs_metadata_state.json")
ABS_DISABLE_CACHE_ENV = os.environ.get("ABS_DISABLE_CACHE", "0")
DEFAULT_DISABLE_CACHE = ABS_DISABLE_CACHE_ENV.lower() in ("1", "true", "yes", "y", "on")


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

    # Использование FantLab
    parser.set_defaults(use_fantlab=DEFAULT_USE_FANTLAB)
    parser.add_argument(
        "--use-fantlab",
        dest="use_fantlab",
        action="store_true",
        help=("После исправления метаданных пробовать автоматически "
              "подтянуть данные книги с провайдера FantLab через /api/search/books. "
              f"По умолчанию: {'включён' if DEFAULT_USE_FANTLAB else 'выключен'}, "
              "значение можно задать через ABS_USE_FANTLAB=0/1/true/false"),
    )
    parser.add_argument(
        "--no-use-fantlab",
        dest="use_fantlab",
        action="store_false",
        help="Отключить использование FantLab для текущего запуска.",
    )
    parser.add_argument(
        "--fantlab-provider",
        default="fantlab",
        help=("Имя провайдера метаданных для FantLab "
              "(как настроено в Audiobookshelf, по умолчанию: 'fantlab')"),
    )

    # Кеш состояния
    parser.add_argument(
        "--cache-file",
        default=ABS_CACHE_FILE_ENV,
        help=("Путь к json-файлу состояния. "
              f"По умолчанию: {ABS_CACHE_FILE_ENV!r} (ABS_CACHE_FILE)"),
    )
    parser.set_defaults(disable_cache=DEFAULT_DISABLE_CACHE)
    parser.add_argument(
        "--disable-cache",
        dest="disable_cache",
        action="store_true",
        help=("Отключить кеширование для текущего запуска "
              "(или ABS_DISABLE_CACHE=1)."),
    )
    parser.add_argument(
        "--enable-cache",
        dest="disable_cache",
        action="store_false",
        help="Явно включить кеширование для текущего запуска.",
    )

    return parser.parse_args()


def chunked(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_session(base_url: str, token: Optional[str]) -> Tuple[requests.Session, str]:
    base_url = base_url.rstrip("/")
    session = requests.Session()
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session, base_url


# ======================================================================
# КЕШ СОСТОЯНИЯ
# ======================================================================

def load_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("items"), dict):
            data.setdefault("version", 1)
            return data
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return {"version": 1, "items": {}}


def save_state(path: str, state: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="abs_state_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def get_item_state(state: Dict[str, Any], item_id: str) -> Dict[str, Any]:
    items = state.setdefault("items", {})
    item_state = items.get(item_id)
    if not isinstance(item_state, dict):
        item_state = {}
        items[item_id] = item_state
    return item_state


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


# --- Работа с API Audiobookshelf ------------------------------------------


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
    if not metadata_updates:
        return

    payload = {"metadata": metadata_updates}
    if dry_run:
        # Ничего не отправляем — только логика просмотра.
        return

    resp = session.patch(f"{base_url}/api/items/{item_id}/media", json=payload)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"  !!! Ошибка PATCH /items/{item_id}/media: {e} — {resp.text}")


# --- FantLab через /api/search/books --------------------------------------


def _normalize_text(s: str) -> str:
    """Упрощаем строку для грубого сравнения (убираем знаки, приводим к lower)."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^0-9a-zа-яё\s]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_trailing_punct_for_search(title: str) -> str:
    """
    Убираем хвостовые знаки препинания в конце названия
    для поискового запроса на FantLab.
    Пример: 'Понедельник начинается в субботу!' -> 'Понедельник начинается в субботу'
    """
    if not title:
        return ""
    title = title.strip()
    title = re.sub(r"[!?.…,:;]+$", "", title).strip()
    return title


def search_fantlab_book(
    session: requests.Session,
    base_url: str,
    title: str,
    author: Optional[str],
    provider: str = "fantlab",
) -> Optional[Dict[str, Any]]:
    """
    Ищем книгу в FantLab через:
      GET /api/search/books?title=...&author=...&provider=fantlab

    Делаем две попытки:
      1) title + author (если author не пустой)
      2) только title (author="")
    Возвращаем один «лучший» результат (сейчас просто первый из ответа).
    """
    original_title = (title or "").strip()

    query_title = strip_trailing_punct_for_search(original_title) or original_title
    author = (author or "").strip()

    if not query_title:
        return None

    def _do_request(params: Dict[str, str]) -> List[Dict[str, Any]]:
        resp = session.get(f"{base_url}/api/search/books", params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []

    results: List[Dict[str, Any]] = []
    if author:
        params1 = {
            "title": query_title,
            "author": author,
            "provider": provider,
        }
        try:
            results = _do_request(params1)
        except Exception as e:
            print(f"  [FantLab] Ошибка запроса (title+author): {e}")

    if not results:
        params2 = {
            "title": query_title,
            "provider": provider,
        }
        try:
            results = _do_request(params2)
        except Exception as e:
            print(f"  [FantLab] Ошибка запроса (title only): {e}")
            results = []

    if not results:
        return None

    best = results[0]

    norm_q = _normalize_text(original_title or query_title)
    norm_res = _normalize_text(str(best.get("title", "")))
    if norm_q and norm_res and norm_q not in norm_res and norm_res not in norm_q:
        print(
            f"  [FantLab] Внимание: название из FantLab '{best.get('title')}' "
            f"сильно отличается от запроса '{original_title or query_title}'"
        )

    return best


def build_fantlab_metadata_updates(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Строим dict для PATCH /items/<ID>/media/metadata на основе результата
    FantLab. НЕ трогаем title и обложку.
    """
    updates: Dict[str, Any] = {}

    subtitle = result.get("subtitle")
    if subtitle:
        updates["subtitle"] = subtitle

    desc = result.get("description")
    if desc:
        updates["description"] = desc

    publisher = result.get("publisher")
    if publisher:
        updates["publisher"] = publisher

    year = result.get("publishedYear")
    if year:
        updates["publishedYear"] = str(year)

    isbn = result.get("isbn")
    if isbn:
        updates["isbn"] = isbn

    genres = result.get("genres") or []
    if isinstance(genres, list) and genres:
        updates["genres"] = genres

    return updates


# --- Обложки ---------------------------------------------------------------

def get_image_size_from_bytes(data: bytes) -> Optional[Tuple[int, int]]:
    if not data or Image is None:
        return None
    try:
        with Image.open(BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def get_image_size_from_url(session: requests.Session, url: str, timeout: int = 20) -> Optional[Tuple[int, int]]:
    if not url or Image is None:
        return None
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return get_image_size_from_bytes(r.content)
    except Exception:
        return None


def get_current_item_cover_size(
    session: requests.Session,
    base_url: str,
    item_id: str,
    timeout: int = 20,
) -> Optional[Tuple[int, int]]:
    try:
        r = session.get(
            f"{base_url}/api/items/{item_id}/cover",
            params={"raw": 1},
            timeout=timeout,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return get_image_size_from_bytes(r.content)
    except Exception:
        return None


def download_cover_from_url_to_item(
    session: requests.Session,
    base_url: str,
    item_id: str,
    cover_url: str,
    dry_run: bool = True,
) -> bool:
    if not cover_url:
        return False

    if dry_run:
        print(f"  [DRY-RUN][Cover] POST /api/items/{item_id}/cover url={cover_url}")
        return True

    try:
        r = session.post(
            f"{base_url}/api/items/{item_id}/cover",
            json={"url": cover_url},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json() if r.headers.get("Content-Type", "").startswith("application/json") else {}
        if data.get("success") is False:
            print(f"  [Cover] ABS вернул success=false для {item_id}")
            return False
        return True
    except Exception as e:
        print(f"  [Cover] Ошибка загрузки обложки для {item_id}: {e}")
        return False


def maybe_update_cover_from_fantlab(
    session: requests.Session,
    base_url: str,
    item_id: str,
    fantlab_cover_url: Optional[str],
    existing_cover_path: Optional[str] = None,
    dry_run: bool = True,
) -> None:
    if not fantlab_cover_url:
        return

    has_cover = bool(existing_cover_path)

    if Image is None:
        if not has_cover:
            print("  [Cover] Pillow не установлен и обложки нет — берем обложку FantLab.")
            download_cover_from_url_to_item(
                session, base_url, item_id, fantlab_cover_url, dry_run=dry_run
            )
        return

    current_size = get_current_item_cover_size(session, base_url, item_id)
    fantlab_size = get_image_size_from_url(session, fantlab_cover_url)

    if not has_cover:
        print("  [Cover] У книги нет обложки — берем обложку FantLab.")
        download_cover_from_url_to_item(
            session, base_url, item_id, fantlab_cover_url, dry_run=dry_run
        )
        return

    if current_size is None or fantlab_size is None:
        return

    cw, ch = current_size
    nw, nh = fantlab_size
    if (nw * nh) > (cw * ch) * 1.05:
        print("  [Cover] Обложка FantLab заметно больше — обновляем.")
        download_cover_from_url_to_item(
            session, base_url, item_id, fantlab_cover_url, dry_run=dry_run
        )


# --- Вспомогательное: получить текущие Album/Artist из файлов --------------

def extract_file_album_artist(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    media = item.get("media") or {}
    audio_files = media.get("audioFiles") or []

    meta_tags = None
    for af in audio_files:
        tags = af.get("metaTags") or {}
        if tags.get("tagAlbum"):
            meta_tags = tags
            break

    if not meta_tags:
        return None, None

    return meta_tags.get("tagAlbum") or None, meta_tags.get("tagArtist") or None


# --- Основная логика -------------------------------------------------------

def process_book_item(
    item: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    На вход — libraryItem (из batch/get).
    Возвращаем:
      - dict с изменённым metadata (только изменяемые поля) или None, если ничего не нужно менять
      - dict с информацией для лога + parsed_* для кеша
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
        "parsed_authors": [],
        "parsed_title": None,
    }

    authors_meta = meta.get("authors") or []
    if authors_meta:
        names = [a.get("name") for a in authors_meta if a.get("name")]
        log_info["old_author"] = ", ".join(names) if names else None

    narrators = meta.get("narrators") or []
    if narrators:
        log_info["old_narrator"] = narrators[0]

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
    log_info["parsed_authors"] = authors_from_album
    log_info["parsed_title"] = title_from_album

    if not authors_from_album or not title_from_album:
        return None, log_info

    new_title = title_from_album
    new_authors = authors_from_album
    new_narrator = tag_artist.strip() if tag_artist else None

    updates: Dict[str, Any] = {}

    if new_title and new_title != meta.get("title"):
        updates["title"] = new_title

    new_authors_objs = [{"name": name} for name in new_authors]
    if new_authors_objs:
        updates["authors"] = new_authors_objs

    curr_narrator_name = narrators[0] if narrators else None
    if new_narrator and new_narrator != curr_narrator_name:
        updates["narrators"] = [new_narrator]

    log_info["new_title"] = new_title
    log_info["new_authors"] = ", ".join(new_authors) if new_authors else None
    log_info["new_narrator"] = new_narrator

    return (updates or None), log_info


def main() -> None:
    args = parse_args()
    session, base_url = build_session(args.base_url, args.token)

    use_cache = not args.disable_cache
    state = load_state(args.cache_file) if use_cache else {"version": 1, "items": {}}

    print("Используем настройки:")
    print(f"  base_url       = {base_url!r}")
    print(f"  token          = {'<указан>' if args.token else '<НЕ указан>'}")
    print(f"  dry_run        = {args.dry_run}")
    print(f"  use_fantlab    = {args.use_fantlab}")
    print(f"  fantlab_provider = {args.fantlab_provider!r}")
    print(f"  cache          = {'включен' if use_cache else 'выключен'}")
    if use_cache:
        print(f"  cache_file     = {args.cache_file!r}")
    print()

    if not args.token:
        print("ВНИМАНИЕ: токен не указан (--token или ABS_TOKEN). "
              "Если сервер требует авторизацию, запросы будут падать.\n")

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
        total_fantlab = 0
        total_tags_skipped = 0
        total_fantlab_skipped = 0

        for batch in chunked(item_ids, args.batch_size):
            items = batch_get_items(session, base_url, batch)
            for item in items:
                if item.get("mediaType") != "book":
                    continue

                item_id = item.get("id")
                if not item_id:
                    continue

                item_state = get_item_state(state, item_id) if use_cache else {}

                # Текущее состояние тегов из файлов
                curr_album, curr_artist = extract_file_album_artist(item)

                # Решаем, нужно ли повторно обрабатывать теги
                should_process_tags = True
                if use_cache and item_state.get("tags_done"):
                    if item_state.get("last_tag_album") == curr_album and item_state.get("last_tag_artist") == curr_artist:
                        should_process_tags = False

                updates = None
                info: Dict[str, Any] = {}

                if should_process_tags:
                    updates, info = process_book_item(item)

                    # Применяем обновления по тегам
                    if updates:
                        patch_book_metadata(
                            session,
                            base_url,
                            item_id=item_id,
                            metadata_updates=updates,
                            dry_run=args.dry_run,
                        )
                        total_updates += 1

                    # Если смогли распарсить Album корректно — отмечаем tags_done
                    parsed_title = info.get("parsed_title")
                    parsed_authors = info.get("parsed_authors") or []
                    if use_cache and parsed_title and parsed_authors:
                        item_state["tags_done"] = True
                        item_state["last_tag_album"] = curr_album
                        item_state["last_tag_artist"] = curr_artist
                else:
                    total_tags_skipped += 1

                    # Минимальная info для дальнейшей FantLab-логики
                    media = item.get("media") or {}
                    meta = media.get("metadata") or {}
                    authors_meta = meta.get("authors") or []
                    names = [a.get("name") for a in authors_meta if a.get("name")]
                    old_author = ", ".join(names) if names else None
                    narrators = meta.get("narrators") or []
                    old_narrator = narrators[0] if narrators else None

                    info = {
                        "item_id": item_id,
                        "path": item.get("path"),
                        "old_title": meta.get("title"),
                        "old_author": old_author,
                        "old_narrator": old_narrator,
                        "album": curr_album,
                        "artist_tag": curr_artist,
                    }

                # Если нет ни обновлений по тегам, ни FantLab — пропускаем
                if updates is None and not args.use_fantlab:
                    continue

                total_touched += 1

                # Определяем новые значения для поиска
                new_title = info.get("new_title", info.get("old_title"))
                new_author_str = info.get("new_authors", info.get("old_author"))
                new_narrator = info.get("new_narrator", info.get("old_narrator"))

                print(f"\nКнига: {info.get('path')}")
                print(f"  ID: {item_id}")
                print(f"  Album (из файла): {info.get('album')}")
                print(f"  Artist (из файла, чтец?): {info.get('artist_tag')}")
                print(f"  Старый title: {info.get('old_title')!r}")
                print(f"  Старый author: {info.get('old_author')!r}")
                print(f"  Старый narrator: {info.get('old_narrator')!r}")
                print(f"  Новый title: {new_title!r}")
                print(f"  Новый author: {new_author_str!r}")
                print(f"  Новый narrator: {new_narrator!r}")

                # ---------------- FantLab с кешем ----------------
                if args.use_fantlab:
                    should_try_fantlab = True
                    if use_cache and item_state.get("fantlab_applied") is True:
                        should_try_fantlab = False

                    if not should_try_fantlab:
                        total_fantlab_skipped += 1
                    else:
                        search_title = new_title or info.get("old_title") or ""
                        search_author = new_author_str or info.get("old_author")

                        fl_result = search_fantlab_book(
                            session,
                            base_url,
                            title=search_title,
                            author=search_author,
                            provider=args.fantlab_provider,
                        )

                        # Счетчик попыток
                        if use_cache:
                            item_state["fantlab_attempts"] = int(item_state.get("fantlab_attempts") or 0) + 1
                            item_state["fantlab_last_title"] = search_title
                            item_state["fantlab_last_author"] = search_author

                        if not fl_result:
                            print("  [FantLab] Ничего не найдено для этого заголовка.")
                            if use_cache:
                                item_state["fantlab_applied"] = False
                        else:
                            print(
                                "  [FantLab] Найдено соответствие: "
                                f"{fl_result.get('title')!r} — {fl_result.get('author')!r}, "
                                f"год: {fl_result.get('publishedYear')}, "
                                f"жанры: {fl_result.get('genres')}"
                            )
                            fl_updates = build_fantlab_metadata_updates(fl_result)
                            if fl_updates:
                                print(f"  [FantLab] Обновляем поля: {list(fl_updates.keys())}")
                                patch_book_metadata(
                                    session,
                                    base_url,
                                    item_id=item_id,
                                    metadata_updates=fl_updates,
                                    dry_run=args.dry_run,
                                )
                                total_fantlab += 1
                            else:
                                print("  [FantLab] В результате нет полей, которые нужно применить.")

                            # Обложка
                            media = item.get("media") or {}
                            existing_cover_path = media.get("coverPath")

                            maybe_update_cover_from_fantlab(
                                session=session,
                                base_url=base_url,
                                item_id=item_id,
                                fantlab_cover_url=fl_result.get("cover"),
                                existing_cover_path=existing_cover_path,
                                dry_run=args.dry_run,
                            )

                            if use_cache:
                                # Считаем, что FantLab применён, если совпадение найдено,
                                # даже если updates пустые: иначе будет дергать каждый раз.
                                item_state["fantlab_applied"] = True

        print(f"\nИтого по библиотеке {lib_name!r}:")
        print(f"  Всего книг: {len(item_ids)}")
        print(f"  Книг обработано (теги и/или FantLab): {total_touched}")
        print(f"  Пропуск обработки тегов по кешу: {total_tags_skipped}")
        print(f"  PATCH по тегам (или был бы в dry-run): {total_updates}")
        if args.use_fantlab:
            print(f"  Пропуск FantLab по кешу: {total_fantlab_skipped}")
            print(f"  Книг, для которых применены метаданные FantLab: {total_fantlab}")
        print("")

    # Сохраняем состояние только при реальном запуске
    if use_cache and not args.dry_run:
        save_state(args.cache_file, state)


if __name__ == "__main__":
    main()
