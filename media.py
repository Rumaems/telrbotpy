import logging
import os
from urllib.parse import quote

import aiohttp

log = logging.getLogger("flow-ai-kpc")

IMAGE_GEN = os.getenv("IMAGE_GEN", "1") == "1"
# Бесплатный генератор без ключа. Можно заменить на любой другой сервис,
# вернув из функции байты картинки.
IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}?width=1024&height=1024&nologo=true"


async def generate_image(prompt: str):
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
