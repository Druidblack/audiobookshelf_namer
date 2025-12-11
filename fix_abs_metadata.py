#!/usr/bin/env python3
import argparse
import os
import re
import json
import tempfile
import time
from typing import List, Tuple, Dict, Any, Optional
from io import BytesIO

import requests

try:
    from PIL import Image
except ImportError:
    Image = None



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

ABS_RUN_INTERVAL_MIN_ENV = os.environ.get("ABS_RUN_INTERVAL_MINUTES", "60")
try:
    DEFAULT_RUN_INTERVAL_MIN = int(ABS_RUN_INTERVAL_MIN_ENV)
except ValueError:
    DEFAULT_RUN_INTERVAL_MIN = 0

ABS_WAIT_SCAN_ENV = os.environ.get("ABS_WAIT_SCAN", "1")
ABS_WAIT_SCAN_ENABLED = ABS_WAIT_SCAN_ENV.lower() in ("1", "true", "yes", "y", "on")

ABS_SCAN_CHECK_INTERVAL_ENV = os.environ.get("ABS_SCAN_CHECK_INTERVAL", "10")
try:
    ABS_SCAN_CHECK_INTERVAL = max(1, int(ABS_SCAN_CHECK_INTERVAL_ENV))
except ValueError:
    ABS_SCAN_CHECK_INTERVAL = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Исправление метаданных книг в Audiobookshelf "
                    "на основе тегов аудиофайлов + поиск метаданных у провайдеров"
    )
    parser.add_argument(
        "--base-url",
        default=ABS_BASE_URL_ENV,
        help="Базовый URL Audiobookshelf "
             f"(по умолчанию: {ABS_BASE_URL_ENV!r} или ABS_BASE_URL)",
    )
    parser.add_argument(
        "--token",
        default=ABS_TOKEN_ENV,
        help="Bearer-токен Audiobookshelf (по умолчанию из ABS_TOKEN)",
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
        help=("Только показывать изменения (без PATCH). "
              f"По умолчанию: {'включён' if DEFAULT_DRY_RUN else 'выключен'} "
              "(ABS_DRY_RUN)"),
    )

    # Использование цепочки провайдеров
    parser.set_defaults(use_fantlab=DEFAULT_USE_FANTLAB)
    parser.add_argument(
        "--use-fantlab",
        dest="use_fantlab",
        action="store_true",
        help=("Включить поиск метаданных по провайдерам "
              "FantLab → Google Books → Audible. "
              f"По умолчанию: {'включён' if DEFAULT_USE_FANTLAB else 'выключен'} "
              "(ABS_USE_FANTLAB)"),
    )
    parser.add_argument(
        "--no-use-fantlab",
        dest="use_fantlab",
        action="store_false",
        help="Отключить использование провайдеров для текущего запуска.",
    )

    parser.add_argument(
        "--fantlab-provider",
        default="fantlab",
        help="Имя провайдера метаданных FantLab в ABS (по умолчанию: 'fantlab')",
    )
    parser.add_argument(
        "--google-provider",
        default="google",
        help="Имя провайдера метаданных Google Books в ABS (по умолчанию: 'google')",
    )
    parser.add_argument(
        "--audible-provider",
        default="audible",
        help="Имя провайдера метаданных Audible в ABS (по умолчанию: 'audible')",
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
        help="Отключить кеширование для текущего запуска (или ABS_DISABLE_CACHE=1).",
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

# ======================================================================
# ОЖИДАНИЕ СКАНИРОВАНИЯ БИБЛИОТЕКИ В ABS
# ======================================================================

def abs_has_running_tasks(session: requests.Session, base_url: str) -> bool:
    """
    Возвращает True, если в ABS сейчас есть активные задачи (в первую очередь
    сканирование библиотек). Основано на неофициальном эндпоинте /api/tasks.
    Если эндпоинт недоступен, функция тихо возвращает False.
    """
    if not ABS_WAIT_SCAN_ENABLED:
        return False

    url = f"{base_url.rstrip('/')}/api/tasks"
    try:
        resp = session.get(url, timeout=10)
    except Exception as e:
        print(f"[ABS scan] Не удалось запросить /api/tasks: {e}")
        return False

    if resp.status_code != 200:
        print(
            f"[ABS scan] /api/tasks вернул статус {resp.status_code}. "
            "Проверка сканирования будет пропущена для этого запроса.",
        )
        return False

    try:
        data = resp.json()
    except ValueError:
        print("[ABS scan] Ответ /api/tasks не является JSON. Пропускаем проверку.")
        return False

    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return False

    # /api/tasks обычно содержит только активные задачи
    return len(tasks) > 0


def wait_while_abs_scanning(session: requests.Session, base_url: str) -> None:
    """
    Блокирующее ожидание, пока в ABS есть активные задачи.
    Если ABS_WAIT_SCAN отключён, функция ничего не делает.
    """
    if not ABS_WAIT_SCAN_ENABLED:
        return

    while abs_has_running_tasks(session, base_url):
        print(
            f"[ABS scan] Обнаружено сканирование/фоновая задача в ABS. "
            f"Ждём {ABS_SCAN_CHECK_INTERVAL} сек...",
        )
        try:
            time.sleep(ABS_SCAN_CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\nОжидание сканирования прервано пользователем (Ctrl+C). Выход.")
            raise



# ======================================================================
# ПАРСИНГ АВТОРОВ/НАЗВАНИЯ ИЗ ALBUM
# ======================================================================

def _split_authors(author_part: str) -> List[str]:
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
    """
    if not album:
        return [], None

    album = album.strip()

    # 1) «Название»
    open_q = album.find("«")
    close_q = album.rfind("»")
    if open_q != -1 and close_q != -1 and close_q > open_q + 1:
        author_part = album[:open_q].strip()
        title = album[open_q + 1:close_q].strip()
        author_part = re.sub(r"[.:,\-–—]+$", "", author_part).strip()
        authors = _split_authors(author_part)
        return authors, (title or None)

    # 2) "Название"
    open_q = album.find('"')
    close_q = album.rfind('"')
    if open_q != -1 and close_q != -1 and close_q > open_q + 1:
        author_part = album[:open_q].strip()
        title = album[open_q + 1:close_q].strip()
        author_part = re.sub(r"[.:,\-–—]+$", "", author_part).strip()
        authors = _split_authors(author_part)
        return authors, (title or None)

    # 3) 'Автор - Название'
    if " - " in album:
        author_part, title = album.split(" - ", 1)
        authors = _split_authors(author_part)
        return authors, (title.strip() or None)

    # 4) 'Автор — Название'
    for dash in (" — ", " – "):
        if dash in album:
            author_part, title = album.split(dash, 1)
            authors = _split_authors(author_part)
            return authors, (title.strip() or None)

    return [], None


# ======================================================================
# API AUDIOBOOKSHELF
# ======================================================================

def get_book_libraries(
    session: requests.Session,
    base_url: str,
    only_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    resp = session.get(f"{base_url}/api/libraries")
    resp.raise_for_status()
    data = resp.json()
    libs = data.get("libraries", [])

    result = []
    for lib in libs:
        if only_id and lib.get("id") != only_id:
            continue
        if lib.get("mediaType") == "book":
            result.append(lib)
    return result


def get_library_item_ids(
    session: requests.Session,
    base_url: str,
    library_id: str,
) -> List[str]:
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
    if not metadata_updates:
        return

    payload = {"metadata": metadata_updates}
    if dry_run:
        return

    resp = session.patch(f"{base_url}/api/items/{item_id}/media", json=payload)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"  !!! Ошибка PATCH /items/{item_id}/media: {e} — {resp.text}")

# ======================================================================
# НОВОЕ: тег для книг без найденных метаданных
# ======================================================================

NO_METADATA_TAG = "нет метаданных"

def patch_book_tags(
    session: requests.Session,
    base_url: str,
    item_id: str,
    tags: List[str],
    dry_run: bool = True,
) -> None:
    if tags is None:
        return

    payload = {"tags": tags}

    if dry_run:
        print(f"  [DRY-RUN][ABS Tags] PATCH /api/items/{item_id}/media tags={tags}")
        return

    resp = session.patch(f"{base_url}/api/items/{item_id}/media", json=payload)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"  !!! Ошибка PATCH tags /items/{item_id}/media: {e} — {resp.text}")


def ensure_no_metadata_tag(
    session: requests.Session,
    base_url: str,
    item: Dict[str, Any],
    dry_run: bool = True,
) -> bool:
    item_id = item.get("id")
    if not item_id:
        return False

    media = item.get("media") or {}
    current = media.get("tags") or []
    if not isinstance(current, list):
        current = []

    if NO_METADATA_TAG in current:
        return False

    new_tags = current + [NO_METADATA_TAG]
    print(f"  [ABS Tags] Добавляем тег {NO_METADATA_TAG!r}")
    patch_book_tags(session, base_url, item_id, new_tags, dry_run=dry_run)
    return True

# ======================================================================
# НОВОЕ: если метаданные найдены у провайдера — считаем, что теги уже разобраны
# ======================================================================

def mark_tags_done_by_provider(item_state: Dict[str, Any],
                               curr_album: Optional[str],
                               curr_artist: Optional[str]) -> None:
    """
    Помечаем книгу как уже обработанную по тегам,
    чтобы при следующих запусках не тратить время на повторный разбор.
    """
    item_state["tags_done"] = True
    item_state["last_tag_album"] = curr_album
    item_state["last_tag_artist"] = curr_artist
    item_state["tags_done_reason"] = "provider"


# ======================================================================
# ОЧИСТКА/НОРМАЛИЗАЦИЯ НАЗВАНИЙ
# ======================================================================

def remove_radioplay_marker(title: str) -> str:
    if not title:
        return ""
    return re.sub(r"\(\s*радиоспектакль\s*\)", "", title, flags=re.IGNORECASE).strip()


def remove_trailing_story_marker(title: str) -> str:
    """
    Убираем хвостовые ', рассказ' / ', рассказы'
    """
    if not title:
        return ""
    return re.sub(r"\s*,\s*рассказ(?:ы)?\.?\s*$", "", title, flags=re.IGNORECASE).strip()


def remove_trailing_new_translation_marker(title: str) -> str:
    """
    Убираем хвостовой маркер '(новый перевод)'
    """
    if not title:
        return ""
    return re.sub(r"\(\s*новый\s+перевод\s*\)\s*$", "", title, flags=re.IGNORECASE).strip()


def cleanup_title_markers(title: str) -> str:
    title = title or ""
    title = remove_radioplay_marker(title)
    title = remove_trailing_story_marker(title)
    title = remove_trailing_new_translation_marker(title)
    return title.strip()


def strip_trailing_punct_for_search(title: str) -> str:
    """
    Подготовка названия для поиска на провайдерах.
    """
    if not title:
        return ""
    title = cleanup_title_markers(title)
    title = re.sub(r"\s{2,}", " ", title).strip()
    title = re.sub(r"[!?.…,:;]+$", "", title).strip()
    return title


def _normalize_text(s: str) -> str:
    s = cleanup_title_markers(s or "")
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-zа-яё\s]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ======================================================================
# НОВОЕ: эквивалентность "е/ё" для поиска и проверки названий
# ======================================================================

def _yo_to_e(s: str) -> str:
    if not s:
        return ""
    return s.replace("ё", "е").replace("Ё", "Е")


def is_title_compatible_yo_equiv(reference_title: str, candidate_title: str) -> bool:
    """
    Обертка над существующей is_title_compatible.
    Ничего не меняем внутри старой функции — только подаем нормализованные строки.
    """
    return is_title_compatible(_yo_to_e(reference_title), _yo_to_e(candidate_title))


def provider_result_is_safe_variative_yo_equiv(reference_title: str,
                                               search_author: Optional[str],
                                               result: Dict[str, Any],
                                               label: str) -> bool:
    """
    Обертка над существующей provider_result_is_safe_variative.
    Внутренние функции не трогаем.
    """
    if not result:
        return False

    ref2 = _yo_to_e(reference_title)

    # копия результата с нормализованным title
    res2 = dict(result)
    t = res2.get("title")
    if isinstance(t, str):
        res2["title"] = _yo_to_e(t)

    return provider_result_is_safe_variative(ref2, search_author, res2, label)


def search_book_yo_equiv(session: requests.Session,
                         base_url: str,
                         title: str,
                         author: Optional[str],
                         provider: str) -> Optional[Dict[str, Any]]:
    """
    Обертка над существующей search_fantlab_book:
    пробуем исходный title и вариант с ё→е.
    """
    title = title or ""
    variants = [title]
    t2 = _yo_to_e(title)
    if t2 and t2 != title:
        variants.append(t2)

    for t in variants:
        res = search_fantlab_book(session, base_url, title=t, author=author, provider=provider)
        if res:
            return res
    return None


# ======================================================================
# ЗАЩИТА ОТ НЕПРАВИЛЬНЫХ СОВПАДЕНИЙ ПО НАЗВАНИЮ
# ======================================================================

_TITLE_STOPWORDS = {
    "книга", "том", "часть", "серия", "цикл",
    "book", "volume", "part", "series"
}


def _title_word_tokens(s: str) -> List[str]:
    s = cleanup_title_markers(s or "")
    s = s.lower()
    words = re.findall(r"[a-zа-яё]+", s, flags=re.IGNORECASE)
    cleaned = []
    for w in words:
        w = w.lower().strip()
        if len(w) < 2:
            continue
        if w in _TITLE_STOPWORDS:
            continue
        cleaned.append(w)
    return cleaned


def _title_number_tokens(s: str) -> List[str]:
    s = cleanup_title_markers(s or "")
    return re.findall(r"\d+", s)


def is_title_compatible(reference_title: str, candidate_title: str) -> bool:
    """
    Строгая проверка:
    1) Если в reference есть цифры -> в candidate должны быть ТЕ ЖЕ цифры (как множество).
    2) Слова reference должны почти полностью присутствовать в candidate (>= 0.85 покрытия).
    3) Допускаем более мягкое совпадение подстрокой при отсутствии цифр.
    """
    ref = (reference_title or "").strip()
    cand = (candidate_title or "").strip()

    if not ref or not cand:
        return False

    ref_nums = _title_number_tokens(ref)
    cand_nums = _title_number_tokens(cand)

    if ref_nums:
        if not cand_nums:
            return False
        if set(ref_nums) != set(cand_nums):
            return False

    ref_words = _title_word_tokens(ref)
    cand_words = set(_title_word_tokens(cand))

    ref_norm = _normalize_text(ref)
    cand_norm = _normalize_text(cand)

    if len(ref_words) <= 1:
        if ref_norm and cand_norm and (ref_norm in cand_norm or cand_norm in ref_norm):
            return True
        return False

    hit = sum(1 for w in ref_words if w in cand_words)
    coverage = hit / max(1, len(ref_words))

    if coverage >= 0.85:
        return True

    if ref_norm and cand_norm and ref_norm in cand_norm:
        if not ref_nums and coverage >= 0.70:
            return True

    return False


# ======================================================================
# ДОСТАВАНИЕ АВТОРОВ И БАЗОВАЯ (СТАРАЯ) ПРОВЕРКА АВТОРОВ
# ======================================================================
# ВАЖНО: эти функции оставлены как отдельный слой.
# Мы НЕ меняем их логику и добавляем новые вариативные функции ниже.

def _normalize_author_name(name: str) -> str:
    # приводим к нижнему регистру
    name = (name or "").strip().lower()
    # считаем "ё" и "е" одинаковыми
    name = name.replace("ё", "е")
    # выкидываем всё, кроме латиницы/кириллицы и пробелов
    name = re.sub(r"[^a-zа-яё\s]+", " ", name)
    # схлопываем лишние пробелы
    name = re.sub(r"\s+", " ", name)
    return name.strip()



def extract_author_names_from_result(result: Dict[str, Any]) -> List[str]:
    """
    Достаём авторов из результата провайдера:
      - author (str)
      - authors (list[str] или list[dict{name}])
    """
    author_field = result.get("author") or result.get("authors")
    author_names: List[str] = []

    if isinstance(author_field, str):
        author_names = _split_authors(author_field)
    elif isinstance(author_field, list):
        for a in author_field:
            if isinstance(a, dict) and a.get("name"):
                author_names.append(str(a["name"]).strip())
            elif isinstance(a, str):
                author_names.append(a.strip())

    return [n for n in (s.strip() for s in author_names) if n]


def is_author_compatible(search_author: Optional[str], result: Dict[str, Any]) -> bool:
    """
    Требование:
      автор найденной книги должен совпадать с автором, используемым для поиска.

    Реализация:
      - если search_author пустой -> True (не блокируем)
      - сравниваем множества нормализованных авторов;
        считаем совпадением, если все авторы поиска присутствуют в результате
        (подмножество).
    """
    if not search_author:
        return True

    search_list = _split_authors(str(search_author))
    search_set = {_normalize_author_name(a) for a in search_list if _normalize_author_name(a)}
    if not search_set:
        return True

    result_list = extract_author_names_from_result(result)
    result_set = {_normalize_author_name(a) for a in result_list if _normalize_author_name(a)}
    if not result_set:
        return False

    return search_set.issubset(result_set)


def provider_result_is_safe(reference_title: str,
                            search_author: Optional[str],
                            result: Dict[str, Any],
                            label: str) -> bool:
    """
    СТАРАЯ safety-проверка (оставлена без изменения).
    """
    cand_title = str(result.get("title") or "").strip()
    if not cand_title:
        print(f"  [{label}] У результата нет title — пропуск.")
        return False

    ok_title = is_title_compatible(reference_title, cand_title)
    if not ok_title:
        print(
            f"  [{label}] Подозрительное совпадение по названию. "
            f"ABS: {reference_title!r} | {label}: {cand_title!r}. "
            f"Метаданные НЕ применяем."
        )
        return False

    ok_author = is_author_compatible(search_author, result)
    if not ok_author:
        res_auth = result.get("author") or result.get("authors")
        print(
            f"  [{label}] Подозрительное совпадение по автору. "
            f"Ищем: {search_author!r} | {label}: {res_auth!r}. "
            f"Метаданные НЕ применяем."
        )
        return False

    return True


# ======================================================================
# НОВОЕ: ВАРИАТИВНАЯ ПРОВЕРКА АВТОРОВ (ДОБАВЛЕНИЕ БЕЗ ИЗМЕНЕНИЯ СТАРЫХ ФУНКЦИЙ)
# ======================================================================

_PATRONYMIC_SUFFIXES = (
    "ович", "евич", "вич",
    "овна", "евна", "вна",
    "ична", "инич", "инична",
)


def _author_tokens(name: str) -> List[str]:
    """
    Токены имени автора без учета регистра, пунктуации и порядка слов.
    Буквы "е" и "ё" считаем одинаковыми.
    """
    name = (name or "").strip().lower()
    # уравниваем "ё" и "е" перед токенизацией
    name = name.replace("ё", "е")
    tokens = re.findall(r"[a-zа-яё]+", name, flags=re.IGNORECASE)
    return [t for t in tokens if len(t) > 1]



def _is_patronymic_token(token: str) -> bool:
    t = (token or "").lower()
    return any(t.endswith(suf) for suf in _PATRONYMIC_SUFFIXES)


def _core_author_set(tokens: List[str]) -> set:
    """
    Убираем отчества по эвристике суффиксов.
    Если после удаления пусто — возвращаем исходный набор.
    """
    s = set(tokens or [])
    core = {t for t in s if not _is_patronymic_token(t)}
    return core if core else s


def author_name_matches_variative(search_name: str, candidate_name: str) -> bool:
    """
    Вариативное сравнение одного автора:
    - 'Гейман Нил' == 'Нил Гейман'
    - 'Алексей Викторович Иванов' ~ 'Алексей Иванов'
    """
    s_tokens = _author_tokens(search_name)
    c_tokens = _author_tokens(candidate_name)

    if not s_tokens or not c_tokens:
        return False

    s_core = _core_author_set(s_tokens)
    c_core = _core_author_set(c_tokens)

    inter = s_core & c_core
    if not inter:
        return False

    if s_core.issubset(c_core) or c_core.issubset(s_core):
        return True

    if len(s_core) >= 2:
        overlap = len(inter) / max(1, len(s_core))
        return overlap >= 0.8

    only = next(iter(s_core))
    return only in c_core


def is_author_compatible_variative(search_author: Optional[str], result: Dict[str, Any]) -> bool:
    """
    Вариативная проверка списка авторов книги:
    - порядок соавторов не важен;
    - каждый автор из search должен найти соответствие в result.
    """
    if not search_author:
        return True

    search_list = _split_authors(str(search_author))
    search_list = [a for a in (s.strip() for s in search_list) if a]
    if not search_list:
        return True

    result_list = extract_author_names_from_result(result)
    if not result_list:
        return False

    for s_name in search_list:
        if not any(author_name_matches_variative(s_name, c_name) for c_name in result_list):
            return False

    return True


def provider_result_is_safe_variative(reference_title: str,
                                      search_author: Optional[str],
                                      result: Dict[str, Any],
                                      label: str) -> bool:
    """
    Новый вариант safety-проверки:
    - проверка названия остается прежней (is_title_compatible)
    - проверка автора — вариативная
    """
    cand_title = str(result.get("title") or "").strip()
    if not cand_title:
        print(f"  [{label}] У результата нет title — пропуск.")
        return False

    ok_title = is_title_compatible(reference_title, cand_title)
    if not ok_title:
        print(
            f"  [{label}] Подозрительное совпадение по названию. "
            f"ABS: {reference_title!r} | {label}: {cand_title!r}. "
            f"Метаданные НЕ применяем."
        )
        return False

    ok_author = is_author_compatible_variative(search_author, result)
    if not ok_author:
        res_auth = result.get("author") or result.get("authors")
        print(
            f"  [{label}] Подозрительное совпадение по автору. "
            f"Ищем: {search_author!r} | {label}: {res_auth!r}. "
            f"Метаданные НЕ применяем."
        )
        return False

    return True


# ======================================================================
# ПОИСК КНИГ У ПРОВАЙДЕРОВ ЧЕРЕЗ ABS
# ======================================================================

def search_fantlab_book(
    session: requests.Session,
    base_url: str,
    title: str,
    author: Optional[str],
    provider: str = "fantlab",
) -> Optional[Dict[str, Any]]:
    """
    Универсальный поиск через:
      GET /api/search/books?title=...&author=...&provider=<provider>

    1) title+author
    2) title
    Возвращаем первый результат.
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
        params1 = {"title": query_title, "author": author, "provider": provider}
        try:
            results = _do_request(params1)
        except Exception as e:
            print(f"  [{provider}] Ошибка запроса (title+author): {e}")

    if not results:
        params2 = {"title": query_title, "provider": provider}
        try:
            results = _do_request(params2)
        except Exception as e:
            print(f"  [{provider}] Ошибка запроса (title only): {e}")
            results = []

    if not results:
        return None

    return results[0]


# ======================================================================
# ПРИМЕНЕНИЕ МЕТАДАННЫХ ПРОВАЙДЕРА
# ======================================================================

def build_fantlab_metadata_updates(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Унифицированный сбор метаданных из результата /api/search/books
    (FantLab/Google Books/Audible и др.).
    """
    updates: Dict[str, Any] = {}

    fl_title = result.get("title")
    if fl_title:
        updates["title"] = fl_title

    author_names = extract_author_names_from_result(result)
    if author_names:
        updates["authors"] = [{"name": name} for name in author_names]

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


# ======================================================================
# ОБЛОЖКИ
# ======================================================================

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
    provider_cover_url: Optional[str],
    existing_cover_path: Optional[str] = None,
    dry_run: bool = True,
) -> None:
    if not provider_cover_url:
        return

    has_cover = bool(existing_cover_path)

    if Image is None:
        if not has_cover:
            print("  [Cover] Pillow не установлен и обложки нет — берем обложку провайдера.")
            download_cover_from_url_to_item(
                session, base_url, item_id, provider_cover_url, dry_run=dry_run
            )
        return

    current_size = get_current_item_cover_size(session, base_url, item_id)
    provider_size = get_image_size_from_url(session, provider_cover_url)

    if not has_cover:
        print("  [Cover] У книги нет обложки — берем обложку провайдера.")
        download_cover_from_url_to_item(
            session, base_url, item_id, provider_cover_url, dry_run=dry_run
        )
        return

    if current_size is None or provider_size is None:
        return

    cw, ch = current_size
    nw, nh = provider_size
    if (nw * nh) > (cw * ch) * 1.05:
        print("  [Cover] Обложка провайдера заметно больше — обновляем.")
        download_cover_from_url_to_item(
            session, base_url, item_id, provider_cover_url, dry_run=dry_run
        )


# ======================================================================
# ВСПОМОГАТЕЛЬНОЕ: ALBUM/ARTIST ИЗ ФАЙЛОВ
# ======================================================================

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


# ======================================================================
# ОСНОВНАЯ ЛОГИКА: ИСПРАВЛЕНИЕ ПО ТЕГАМ
# ======================================================================

def process_book_item(
    item: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    На вход — libraryItem (из batch/get).
    Возвращаем:
      - dict с изменённым metadata (только изменяемые поля) или None
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


# ======================================================================
# ОСНОВНОЙ ПРОХОД: ТЕГИ + ПРОВАЙДЕРЫ
# ======================================================================

def run_once(args: argparse.Namespace,
             session: requests.Session,
             base_url: str,
             state: Dict[str, Any],
             use_cache: bool) -> None:
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
        # На входе в каждую библиотеку ждём окончания сканирования
        wait_while_abs_scanning(session, base_url)

        lib_id = lib["id"]
        lib_name = lib.get("name")
        print(f"=== Библиотека {lib_name!r} ({lib_id}) ===")

        item_ids = get_library_item_ids(session, base_url, lib_id)
        print(f"  Книг в библиотеке: {len(item_ids)}")

        total_updates = 0
        total_touched = 0
        total_fantlab = 0
        total_google = 0
        total_audible = 0
        total_tags_skipped = 0
        total_provider_skipped = 0

        for batch in chunked(item_ids, args.batch_size):
            # Между батчами также проверяем: если ABS запустил сканирование, ждём
            wait_while_abs_scanning(session, base_url)

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

                    if updates:
                        patch_book_metadata(
                            session,
                            base_url,
                            item_id=item_id,
                            metadata_updates=updates,
                            dry_run=args.dry_run,
                        )
                        total_updates += 1

                    parsed_title = info.get("parsed_title")
                    parsed_authors = info.get("parsed_authors") or []
                    if use_cache and parsed_title and parsed_authors:
                        item_state["tags_done"] = True
                        item_state["last_tag_album"] = curr_album
                        item_state["last_tag_artist"] = curr_artist
                else:
                    total_tags_skipped += 1

                    # Минимальная info для провайдеров
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

                if updates is None and not args.use_fantlab:
                    # --- НОВОЕ: сохраняем состояние после книги ---
                    if use_cache and not args.dry_run:
                        try:
                            save_state(args.cache_file, state)
                        except Exception as e:
                            print(f"  [State] Ошибка сохранения состояния после книги {item_id}: {e}")
                    continue

                total_touched += 1

                # Значения для поиска
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

                # Reference title для защиты
                reference_title = (new_title or info.get("old_title") or "").strip()

                # ---------------- FantLab -> Google -> Audible ----------------
                if args.use_fantlab:
                    should_try_provider = True
                    if use_cache and item_state.get("fantlab_applied") is True:
                        should_try_provider = False

                    if not should_try_provider:
                        total_provider_skipped += 1
                        # --- НОВОЕ: сохраняем состояние после книги ---
                        if use_cache and not args.dry_run:
                            try:
                                save_state(args.cache_file, state)
                            except Exception as e:
                                print(f"  [State] Ошибка сохранения состояния после книги {item_id}: {e}")
                        continue

                    search_title = new_title or info.get("old_title") or ""
                    search_author = new_author_str or info.get("old_author")

                    # 1) FantLab
                    fl_result = search_book_yo_equiv(
                        session, base_url,
                        title=search_title,
                        author=search_author,
                        provider=args.fantlab_provider,
                    )

                    if use_cache:
                        item_state["fantlab_attempts"] = int(item_state.get("fantlab_attempts") or 0) + 1
                        item_state["fantlab_last_title"] = search_title
                        item_state["fantlab_last_author"] = search_author

                    if fl_result and not provider_result_is_safe_variative_yo_equiv(reference_title, search_author, fl_result, "FantLab"):
                        fl_result = None

                    if not fl_result:
                        print("  [FantLab] Ничего безопасного не найдено. Пробуем Google Books...")

                        # 2) Google Books
                        gb_result = search_book_yo_equiv(
                            session, base_url,
                            title=search_title,
                            author=search_author,
                            provider=args.google_provider,
                        )

                        if gb_result and not provider_result_is_safe_variative_yo_equiv(reference_title, search_author, gb_result, "Google"):
                            gb_result = None

                        if not gb_result:
                            print("  [Google] Ничего безопасного не найдено. Пробуем Audible...")

                            # 3) Audible
                            ab_result = search_book_yo_equiv(
                                session, base_url,
                                title=search_title,
                                author=search_author,
                                provider=args.audible_provider,
                            )

                            if ab_result and not provider_result_is_safe_variative_yo_equiv(reference_title, search_author, ab_result, "Audible"):
                                ab_result = None

                            if not ab_result:
                                print("  [Audible] Ничего безопасного не найдено для этого заголовка.")
                                ensure_no_metadata_tag(session, base_url, item, dry_run=args.dry_run)
                                if use_cache:
                                    item_state["fantlab_applied"] = False

                            else:
                                print(
                                    "  [Audible] Найдено соответствие: "
                                    f"{ab_result.get('title')!r} — {ab_result.get('author') or ab_result.get('authors')!r}"
                                )

                                ab_updates = build_fantlab_metadata_updates(ab_result)
                                if ab_updates:
                                    print(f"  [Audible] Обновляем поля: {list(ab_updates.keys())}")
                                    patch_book_metadata(
                                        session,
                                        base_url,
                                        item_id=item_id,
                                        metadata_updates=ab_updates,
                                        dry_run=args.dry_run,
                                    )
                                    total_audible += 1
                                else:
                                    print("  [Audible] В результате нет полей, которые нужно применить.")

                                media = item.get("media") or {}
                                existing_cover_path = media.get("coverPath")

                                maybe_update_cover_from_fantlab(
                                    session=session,
                                    base_url=base_url,
                                    item_id=item_id,
                                    provider_cover_url=ab_result.get("cover"),
                                    existing_cover_path=existing_cover_path,
                                    dry_run=args.dry_run,
                                )

                                if use_cache:
                                    item_state["fantlab_applied"] = True
                                    mark_tags_done_by_provider(item_state, curr_album, curr_artist)

                        else:
                            print(
                                "  [Google] Найдено соответствие: "
                                f"{gb_result.get('title')!r} — {gb_result.get('author') or gb_result.get('authors')!r}"
                            )

                            gb_updates = build_fantlab_metadata_updates(gb_result)
                            if gb_updates:
                                print(f"  [Google] Обновляем поля: {list(gb_updates.keys())}")
                                patch_book_metadata(
                                    session,
                                    base_url,
                                    item_id=item_id,
                                    metadata_updates=gb_updates,
                                    dry_run=args.dry_run,
                                )
                                total_google += 1
                            else:
                                print("  [Google] В результате нет полей, которые нужно применить.")

                            media = item.get("media") or {}
                            existing_cover_path = media.get("coverPath")

                            maybe_update_cover_from_fantlab(
                                session=session,
                                base_url=base_url,
                                item_id=item_id,
                                provider_cover_url=gb_result.get("cover"),
                                existing_cover_path=existing_cover_path,
                                dry_run=args.dry_run,
                            )

                            if use_cache:
                                item_state["fantlab_applied"] = True
                                mark_tags_done_by_provider(item_state, curr_album, curr_artist)

                    else:
                        print(
                            "  [FantLab] Найдено соответствие: "
                            f"{fl_result.get('title')!r} — {fl_result.get('author') or fl_result.get('authors')!r}"
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

                        media = item.get("media") or {}
                        existing_cover_path = media.get("coverPath")

                        maybe_update_cover_from_fantlab(
                            session=session,
                            base_url=base_url,
                            item_id=item_id,
                            provider_cover_url=fl_result.get("cover"),
                            existing_cover_path=existing_cover_path,
                            dry_run=args.dry_run,
                        )

                        if use_cache:
                            item_state["fantlab_applied"] = True
                            mark_tags_done_by_provider(item_state, curr_album, curr_artist)

                # =========================================================
                # НОВОЕ: сохраняем состояние после обработки КАЖДОЙ книги
                # =========================================================
                if use_cache and not args.dry_run:
                    try:
                        save_state(args.cache_file, state)
                    except Exception as e:
                        print(f"  [State] Ошибка сохранения состояния после книги {item_id}: {e}")

        print(f"\nИтого по библиотеке {lib_name!r}:")
        print(f"  Всего книг: {len(item_ids)}")
        print(f"  Книг обработано (теги и/или провайдеры): {total_touched}")
        print(f"  Пропуск обработки тегов по кешу: {total_tags_skipped}")
        print(f"  PATCH по тегам (или был бы в dry-run): {total_updates}")
        if args.use_fantlab:
            print(f"  Пропуск поиска провайдеров по кешу: {total_provider_skipped}")
            print(f"  Применено FantLab: {total_fantlab}")
            print(f"  Применено Google: {total_google}")
            print(f"  Применено Audible: {total_audible}")
        print("")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    args = parse_args()
    session, base_url = build_session(args.base_url, args.token)

    use_cache = not args.disable_cache
    state = load_state(args.cache_file) if use_cache else {"version": 1, "items": {}}

    print("Используем настройки:")
    print(f"  base_url          = {base_url!r}")
    print(f"  token             = {'<указан>' if args.token else '<НЕ указан>'}")
    print(f"  dry_run           = {args.dry_run}")
    print(f"  use_fantlab_chain = {args.use_fantlab}")
    print(f"  fantlab_provider  = {args.fantlab_provider!r}")
    print(f"  google_provider   = {args.google_provider!r}")
    print(f"  audible_provider  = {args.audible_provider!r}")
    print(f"  cache             = {'включен' if use_cache else 'выключен'}")
    if use_cache:
        print(f"  cache_file        = {args.cache_file!r}")
    print(f"  run_interval      = {DEFAULT_RUN_INTERVAL_MIN} мин (ABS_RUN_INTERVAL_MINUTES)")
    print()

    if not args.token:
        print("ВНИМАНИЕ: токен не указан (--token или ABS_TOKEN). "
              "Если сервер требует авторизацию, запросы могут падать.\n")

    interval_min = DEFAULT_RUN_INTERVAL_MIN

    if interval_min <= 0:
        # Одиночный запуск — сначала ждём окончания сканирования
        wait_while_abs_scanning(session, base_url)
        run_once(args, session, base_url, state, use_cache)
        if use_cache and not args.dry_run:
            save_state(args.cache_file, state)
    else:
        print(f"Автоматический режим: скрипт будет выполняться каждые {interval_min} минут.")
        iteration = 0
        while True:
            iteration += 1
            print(f"\n===== Итерация {iteration} =====")
            # Перед каждой итерацией ждём окончания сканирования
            wait_while_abs_scanning(session, base_url)
            run_once(args, session, base_url, state, use_cache)
            if use_cache and not args.dry_run:
                save_state(args.cache_file, state)

            print(f"\nОжидание {interval_min} минут до следующего запуска...")
            try:
                time.sleep(interval_min * 60)
            except KeyboardInterrupt:
                print("\nОстановлено пользователем (Ctrl+C). Выход.")
                break


if __name__ == "__main__":
    main()
