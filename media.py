import logging
import os
import random
import re
from urllib.parse import quote, unquote

import aiohttp

log = logging.getLogger("flow-ai-kpc")

IMAGE_GEN = os.getenv("IMAGE_GEN", "1") == "1"
# Бесплатный генератор без ключа.
IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}?width=1024&height=1024&nologo=true"


async def generate_image(prompt: str):
    """Генерирует картинку по описанию (AI)."""
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


async def search_image(query: str):
    """
    Ищет картинку в интернете по запросу (бесплатно, без ключей).
    Использует DuckDuckGo + fallback Bing.
    Возвращает bytes картинки или None.
    """
    if not query or not query.strip():
        return None
    query = query.strip()[:120]

    try:
        timeout = aiohttp.ClientTimeout(total=25)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as s:
            # 1) DuckDuckGo image search
            async with s.get(f"https://duckduckgo.com/?q={quote(query)}") as r:
                html = await r.text()
                m = re.search(r'vqd=["\']([^"\']+)["\']', html)
                vqd = m.group(1) if m else None

            if not vqd:
                return await _fallback_bing_image(s, query)

            ddg_url = (
                "https://duckduckgo.com/i.js"
                f"?l=us-en&o=json&q={quote(query)}&vqd={quote(vqd)}&f=,,,&p=1"
            )
            async with s.get(ddg_url) as r:
                if r.status != 200:
                    log.warning("DDG image search status %s", r.status)
                    return await _fallback_bing_image(s, query)
                data = await r.json(content_type=None)

            results = data.get("results") or []
            candidates = [item.get("image") for item in results[:8] if item.get("image")]
            if not candidates:
                return await _fallback_bing_image(s, query)

            img_url = random.choice(candidates)
            async with s.get(img_url) as r:
                if r.status != 200:
                    return None
                content_type = r.headers.get("Content-Type", "")
                if not content_type.startswith("image"):
                    return None
                return await r.read()

    except Exception:
        log.exception("Image search error for %r", query)
        return None


async def _fallback_bing_image(session: aiohttp.ClientSession, query: str):
    """Запасной вариант — парсим Bing Images (без ключа)."""
    try:
        url = f"https://www.bing.com/images/search?q={quote(query)}&form=HDRSC2&first=1"
        async with session.get(url) as r:
            html = await r.text()
        matches = re.findall(r'murl&quot;:&quot;(https?://[^&]+?)&quot;', html)
        if not matches:
            matches = re.findall(r'"murl":"(https?://[^"]+)"', html)
        if not matches:
            return None
        img_url = random.choice(matches[:6])
        img_url = unquote(img_url)
        async with session.get(img_url) as r:
            if r.status != 200:
                return None
            if not r.headers.get("Content-Type", "").startswith("image"):
                return None
            return await r.read()
    except Exception:
        log.exception("Bing fallback failed")
        return None
