import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIStatusError, RateLimitError

load_dotenv()

log = logging.getLogger("flow-ai-kpc")


@dataclass
class Provider:
    name: str
    client: AsyncOpenAI
    model: str


def _load_providers() -> List[Provider]:
    """
    Формат AI_PROVIDERS:
    name|api_key|base_url|model;name2|key2|base2|model2
    """
    raw = os.getenv("AI_PROVIDERS", "").strip()
    providers: List[Provider] = []

    if raw:
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                name, key, base_url, model = [x.strip() for x in part.split("|", 3)]
                client = AsyncOpenAI(api_key=key, base_url=base_url)
                providers.append(Provider(name=name, client=client, model=model))
            except Exception as e:
                log.error("Не удалось разобрать провайдера: %s (%s)", part, e)

    # Запасной вариант (старый способ)
    if not providers:
        key = os.getenv("AI_API_KEY")
        if not key:
            raise RuntimeError("Не задан ни AI_PROVIDERS, ни AI_API_KEY")
        base_url = os.getenv("AI_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.getenv("AI_MODEL", "openai/gpt-oss-120b")
        client = AsyncOpenAI(api_key=key, base_url=base_url)
        providers.append(Provider(name="default", client=client, model=model))

    log.info("Загружено провайдеров: %s", [p.name for p in providers])
    return providers


PROVIDERS = _load_providers()
CURRENT_IDX = 0

FAST_MODEL = os.getenv("AI_FAST_MODEL", "openai/gpt-oss-20b")


def _get_provider() -> Provider:
    return PROVIDERS[CURRENT_IDX]


def _switch_provider(reason: str = ""):
    global CURRENT_IDX
    old = PROVIDERS[CURRENT_IDX].name
    CURRENT_IDX = (CURRENT_IDX + 1) % len(PROVIDERS)
    new = PROVIDERS[CURRENT_IDX].name
    log.warning("Переключаю провайдер: %s → %s %s", old, new, f"({reason})" if reason else "")


SYSTEM = """Ты — {bot_name}, живой и естественный участник Telegram-чата.
Разговаривай естественно: замечай контекст, помни собеседников, подхватывай темы,
задавай уместные вопросы и иногда сам начинай разговор.
Не отвечай механически на каждое сообщение. Если ответа не требуется — верни ровно NO_REPLY.
Не говори, что ты ИИ, если тебя прямо не спросили; на прямой вопрос отвечай честно.
Не выдумывай реальный личный опыт.
Пиши без канцелярита и шаблонов. Обычно 1–4 коротких предложения.
Не повторяй одну и ту же мысль. Не начинай каждую реплику с имени человека.
Можно обсуждать игры, мемы, технику, погоду, бытовые темы и темы из чата.
Если человек явно обращается к тебе — отвечай. Если разговор идёт, можешь задать встречный вопрос.
Матерься редко: только если это очень уместно по контексту и настроению. В большинстве ответов обходись без мата и грубой лексики.

Стикеры и картинки (необязательно, используй редко):
- Чтобы кинуть стикер, добавь в конец ответа тег [STICKER:😂] с одним эмодзи по настроению.
  Не чаще чем примерно в одном ответе из пяти. Можно ответить одним стикером без текста.
- Чтобы прислать РЕАЛЬНОЕ фото из интернета (поиск), добавь в конец ответа тег [PHOTO: короткий запрос на английском].
  Обязательно используй, если просят «покажи», «кинь фото», «скинь картинку», «найди фото» чего-то реального
  (животное, машина, еда, место, человек из мема, предмет и т.п.).
  Примеры: [PHOTO: cute cat] [PHOTO: red ferrari] [PHOTO: pizza close up]
- Чтобы СГЕНЕРИРОВАТЬ картинку нейросетью, добавь тег [IMG: подробное описание сцены на английском].
  Только когда явно просят «нарисуй» / «сгенерируй» или нужна фантазия/несуществующее.
- Максимум один тег картинки за ответ (либо PHOTO, либо IMG). Теги не объясняй и не упоминай — они скрыты от людей.
  Можно ответить короткой фразой + тег, например: лови [PHOTO: husky puppy]
Если в сообщении есть фото — посмотри на него и отреагируй по делу, как человек в чате.
Сообщения вида [фото] и [стикер 😂] в истории — это то, что присылали люди."""


def _system(bot_name: str) -> str:
    return SYSTEM.replace("{bot_name}", bot_name)


async def _complete(system: str, content, model: str | None = None, max_tokens: int = 1500) -> str:
    """Делает запрос. При rate-limit переключается на следующий провайдер."""
    last_error = None
    attempts = len(PROVIDERS)

    for _ in range(attempts):
        prov = _get_provider()
        use_model = model or prov.model

        try:
            response = await prov.client.chat.completions.create(
                model=use_model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            )
            return (response.choices[0].message.content or "").strip()

        except (RateLimitError, APIStatusError) as e:
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            last_error = e
            _switch_provider(f"rate limit / {status}")
            continue
        except Exception as e:
            last_error = e
            _switch_provider(f"error: {type(e).__name__}")
            continue

    raise RuntimeError(f"Все провайдеры недоступны. Последняя ошибка: {last_error}") from last_error


def _with_image(text, image):
    if not image:
        return text
    data, media_type = image
    url = f"data:{media_type};base64,{base64.b64encode(data).decode()}"
    return [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": text},
    ]


async def should_speak(context, incoming, direct=False, bot_name="КПК"):
    if direct:
        return True
    prompt = f"""Реши, стоит ли сейчас естественно участвовать в разговоре.
Отвечай только YES или NO.
YES — если есть вопрос, шутка, эмоция, интересная тема или естественный повод добавить короткую реплику.
NO — если сообщение служебное, слишком личное между людьми, односложное или ответ будет спамом.
Не требуй обязательного обращения к боту.

КОНТЕКСТ:
{context}

ПОСЛЕДНЕЕ СООБЩЕНИЕ:
{incoming}"""
    result = await _complete(_system(bot_name), prompt, model=FAST_MODEL, max_tokens=300)
    return result.upper().startswith("YES")


async def chat(context, incoming, direct=False, bot_name="КПК", autonomous=False, image=None):
    if autonomous:
        prompt = f"""КОНТЕКСТ ЧАТА (группа или личка):
{context}

Некоторое время тихо. Сам начни короткую живую реплику, если есть естественный повод.
Можно продолжить недавнюю тему, спросить как дела, пошутить, кинуть мысль по контексту.
Не притворяйся, что тебя только что спросили. Не будь навязчивым.
Если хорошего повода нет — верни ровно NO_REPLY. Если пишешь — 1–3 коротких предложения."""
    else:
        prompt = f"""КОНТЕКСТ ЧАТА:
{context}

ТЕКУЩАЯ СИТУАЦИЯ:
{incoming}

Ответь естественно и по смыслу. Поддерживай разговор: можно продолжить тему,
задать короткий встречный вопрос или отреагировать эмоцией — даже если к тебе
не обращались напрямую, когда это уместно.
Если отвечать действительно не стоит — верни ровно NO_REPLY."""
    return await _complete(_system(bot_name), _with_image(prompt, image))


STICKER_RE = re.compile(r"\[STICKER:\s*([^\]]+?)\s*\]", re.I)
IMG_RE = re.compile(r"\[IMG:\s*([^\]]+?)\s*\]", re.I)
PHOTO_RE = re.compile(r"\[PHOTO:\s*([^\]]+?)\s*\]", re.I)


def parse_reply(raw):
    raw = (raw or "").strip()
    if not raw or "NO_REPLY" in raw:
        return "", [], None, None
    stickers = [s.strip() for s in STICKER_RE.findall(raw)]
    imgs = IMG_RE.findall(raw)
    photos = PHOTO_RE.findall(raw)
    text = PHOTO_RE.sub("", IMG_RE.sub("", STICKER_RE.sub("", raw))).strip()
    gen_prompt = imgs[0].strip() if imgs else None
    search_query = photos[0].strip() if photos else None
    if search_query and gen_prompt:
        gen_prompt = None
    return text, stickers, gen_prompt, search_query


async def extract_facts(context):
    prompt = f"""Извлеки только устойчивые факты, которые помогут в будущих разговорах.
Только явно сказанные факты: имя/как обращаться, хобби, любимые игры, долгосрочные проекты,
устойчивые предпочтения. Не сохраняй пароли, токены, адреса, здоровье, финансы,
политические предпочтения или другие чувствительные данные. Не додумывай.
Верни JSON: [{{"username":"...","fact":"..."}}]. Если ничего нового — [].
Только JSON, без пояснений.

КОНТЕКСТ:
{context}"""
    result = await _complete(
        "Ты аккуратный модуль долговременной памяти. Не выдумывай факты.",
        prompt,
        model=FAST_MODEL,
        max_tokens=1500,
    )
    try:
        text = result.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        return []
