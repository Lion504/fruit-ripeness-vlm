"""Phone-form client pieces for the v6 two-pass pipeline (llama-server, thinking off, strict schemas)."""

import base64
import io
import time

from PIL import Image
from produce_common import prep_image

PASS1_PROMPT = """You are the first step of a produce scanner.
Look at the photo and say which produce item it shows.
Answer with a single JSON object: {"produce": "banana" | "apple" | "tomato" | "none" | "unclear"}
- banana, apple or tomato: the photo clearly shows that fruit.
- none: the photo clearly shows none of these fruits (for example a car, person, shoe, landscape or another fruit).
- unclear: the photo is too blurry, dark or obstructed to tell.
Do not include any text outside the JSON object."""
PASS1_USER = "Which produce is in this photo?"
PASS1_ENUM = ["banana", "apple", "tomato", "none", "unclear"]
PASS1_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "produce_id",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"produce": {"type": "string", "enum": PASS1_ENUM}},
            "required": ["produce"],
            "additionalProperties": False,
        },
    },
}

# stage-only prompts (pass 2), one per fruit add-on.
STAGE_ENUM = {
    "banana": ["unripe", "turning", "ripe", "overripe", "unclear"],
    "apple": ["immature", "semi-mature", "mature", "unclear"],
    "tomato": ["not_yet", "ready", "unclear"],
}
STAGE_DESC = {
    "banana": "Banana ripeness stages: 'unripe', 'turning', 'ripe', 'overripe'.",
    "apple": "Apple ripeness stages: 'immature', 'semi-mature', 'mature'.",
    "tomato": "Tomato ripeness stages: 'not_yet' (green or half-ripe) or 'ready' (fully ripe).",
}


def stage_prompt(fruit):
    return f"""You are the ripeness step of a produce scanner. The photo shows a {fruit}.
{STAGE_DESC[fruit]}
If the photo is too blurry or dark to judge, answer 'unclear'.
Answer with a single JSON object: {{"stage": {" | ".join(repr(x).replace("'", '"') for x in STAGE_ENUM[fruit])}}}
Do not include any text outside the JSON object."""


def stage_format(fruit):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{fruit}_stage",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"stage": {"type": "string", "enum": STAGE_ENUM[fruit]}},
                "required": ["stage"],
                "additionalProperties": False,
            },
        },
    }


def b64_of(path_or_img):
    img = (
        path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    )
    buf = io.BytesIO()
    prep_image(img.convert("RGB")).save(buf, "JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def ask(
    endpoint,
    system,
    user,
    fmt,
    b64,
    lora=None,
    max_tokens=64,
    timeout=75,
    temperature=0.2,
):
    body = {
        "model": "qwen3.5-0.8b",
        "temperature": temperature,
        "seed": 42,
        "max_tokens": max_tokens,
        "response_format": fmt,
        "chat_template_kwargs": {"enable_thinking": False},
        # llama-server reuses the cached prompt start (the system prompt) from earlier requests even when the
        # add-on changed, so answers would depend on which add-on ran before; always compute the full prompt
        "cache_prompt": False,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + b64},
                    },
                ],
            },
        ],
    }
    if lora is not None:
        body["lora"] = lora
    import httpx

    t0 = time.perf_counter()
    r = httpx.post(endpoint + "/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    ms = (time.perf_counter() - t0) * 1000
    j = r.json()
    return j["choices"][0]["message"]["content"], ms, j.get("timings", {})
