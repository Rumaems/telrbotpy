import logging
import os
import random
import re
from urllib.parse import quote

import aiohttp

log = logging.getLogger("flow-ai-kpc")

IMAGE_GEN = os.getenv("IMAGE_GEN", "1") == "1"
# Бесплатный генератор без ключа. Можно заменить на любой другой сервис,
# вернув из функции байты картинки.
IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}?width=1024&height=1024&nologo=true"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


async def generate_image(prompt: str):
    """Генерация картинки (AI)."""
    if not IMAGE_GEN:
        return None
    url = IMAGE_URL.format(prompt=quote(prompt[:500]))
    try:
        timeout = aiohttp.ClientTimeout(total=90)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url) as r:
                if r.status != 200 or not r.headers.get("Content-Type", "").startswith("image"):
                    log.warning("Image gen failed: %s", r.status)
                    return None
                return await r.read()
    except Exception:
        log.exception("Image gen error")
        return None


async def _ddg_vqd(session: aiohttp.ClientSession, query: str) -> str | None:
    """Достаёт vqd-токен со страницы DuckDuckGo."""
    url = f"https://duckduckgo.com/?q={quote(query)}&iax=images&ia=images"
    async with session.get(url) as r:
        if r.status != 200:
            return None
        text = await r.text()
    for pat in (
        r'vqd=["\']([^"\']+)["\']',
        r"vqd=([\d-]+)",
    ):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


async def _ddg_image_urls(session: aiohttp.ClientSession, query: str, limit: int = 8) -> list[str]:
    """Список прямых URL картинок из DuckDuckGo Images."""
    vqd = await _ddg_vqd(session, query)
    if not vqd:
        log.warning("DDG: no vqd for %r", query)
        return []

    params = {
        "l": "us-en",
        "o": "json",
        "q": query,
        "vqd": vqd,
        "f": ",,,",
        "p": "1",
    }
    async with session.get("https://duckduckgo.com/i.js", params=params) as r:
        if r.status != 200:
            log.warning("DDG i.js status %s", r.status)
            return []
        try:
            data = await r.json(content_type=None)
        except Exception:
            log.exception("DDG json parse")
            return []

    urls = []
    for item in data.get("results") or []:
        img = (item.get("image") or "").strip()
        if img.startswith("http"):
            urls.append(img)
        if len(urls) >= limit:
            break
    return urls


async def _download_image(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, allow_redirects=True) as r:
            if r.status != 200:
                return None
            ctype = r.headers.get("Content-Type", "")
            if not ctype.startswith("image"):
                return None
            data = await r.read()
            # отсекаем слишком мелкие/битые
            if len(data) < 3000:
                return None
            return data
    except Exception:
        return None


async def search_image(query: str) -> bytes | None:
    """
    Ищет реальную картинку в интернете через DuckDuckGo и возвращает байты.
    Это не AI-генерация — обычные фото/картинки из выдачи.
    """
    query = (query or "").strip()[:200]
    if not query:
        return None

    timeout = aiohttp.ClientTimeout(total=25)
    headers = {"User-Agent": UA, "Accept": "application/json,text/html,*/*"}
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as s:
            urls = await _ddg_image_urls(s, query, limit=10)
            if not urls:
                log.warning("DDG: no results for %r", query)
                return None
            # берём случайную из первых, чтобы не всегда одну и ту же
            random.shuffle(urls)
            for url in urls[:5]:
                data = await _download_image(s, url)
                if data:
                    return data
            log.warning("DDG: could not download any image for %r", query)
            return None
    except Exception:
        log.exception("Image search error")
        return None
