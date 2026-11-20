"""Prompt, answer schema and data helpers for banana / apple / tomato ripeness.

Stages: banana unripe/turning/ripe/overripe; apple immature/semi-mature/mature;
tomato not_yet (green and half-ripe) / ready (fully ripe).
"""

import os

STAGES = {
    "banana": ["unripe", "turning", "ripe", "overripe"],
    "apple": ["immature", "semi-mature", "mature"],
    "tomato": ["not_yet", "ready"],
}
PRODUCES = list(STAGES)
CONDITIONS = ["no_obvious_defects", "minor_defects", "major_defects", "possible_decay"]
TOMATO_MAP = {"green": "not_yet", "half-ripe": "not_yet", "fully-ripe": "ready"}

PROMPT = """You are an expert produce quality and ripeness assessor.
Your task is to analyze the provided produce photo and produce an accurate visual assessment.

Guidelines:
1. Target Produce: bananas, apples and tomatoes are supported.
   - If the photo clearly shows a banana, an apple or a tomato, set status to 'assessed' and produce to that fruit.
   - If the photo is too blurry, dark, obstructed, or ambiguous to determine, set status to 'unclear'.
   - If the photo clearly contains none of these fruits (e.g. car, person, shoe, landscape), set status to 'unsupported'.

2. Assessment Vocabulary:
   - Banana ripeness: 'unripe', 'turning', 'ripe', 'overripe'
   - Apple ripeness: 'immature', 'semi-mature', 'mature'
   - Tomato ripeness: 'not_yet' (green or half-ripe) or 'ready' (fully ripe)
   - Visible Condition: 'no_obvious_defects', 'minor_defects', 'major_defects', 'possible_decay', or null

3. Describe visible appearance only. Spotting alone does not establish decay.
A photo cannot establish internal quality. Do not give food safety verdicts,
freshness percentages, consumption-time recommendations or buy decisions.

4. Output Format:
   Your final output MUST be a single valid JSON object adhering strictly to this schema:
   {
     "status": "assessed" | "unclear" | "unsupported",
     "produce": "banana" | "apple" | "tomato" | null,
     "ripeness": a stage of the assessed produce (see the vocabulary above) | null,
     "visible_condition": "no_obvious_defects" | "minor_defects" | "major_defects" | "possible_decay" | null,
     "observations": ["observation 1", "observation 2"],
     "selection_note": "A clear, helpful summary note explaining the visible evidence.",
     "limitations": ["Limitation note stating internal quality or firmness cannot be determined from a photo."]
   }
   Do not include any text outside the JSON object.
"""
USER_TEXT = "Assess this produce photo."


def _text():
    return {"type": "string", "minLength": 1, "maxLength": 400}


def schema_format():
    """Strict answer schema: one branch per assessed produce with its own stage enum."""
    shared = {
        "observations": {
            "type": "array",
            "items": _text(),
            "minItems": 1,
            "maxItems": 5,
        },
        "selection_note": _text(),
        "limitations": {
            "type": "array",
            "items": _text(),
            "minItems": 1,
            "maxItems": 5,
        },
    }
    null = {"type": "null"}
    cond = {"type": "string", "enum": CONDITIONS}

    def branch(status, produce, ripeness, condition):
        return {
            "type": "object",
            "properties": {
                "status": {"const": status, "type": "string"},
                "produce": produce,
                "ripeness": ripeness,
                "visible_condition": condition,
                **shared,
            },
            "required": ["status", "produce", "ripeness", "visible_condition", *shared],
            "additionalProperties": False,
        }

    any_produce = {"type": "string", "enum": PRODUCES}
    branches = [
        branch(
            "assessed",
            {"const": p, "type": "string"},
            {"type": "string", "enum": s},
            cond,
        )
        for p, s in STAGES.items()
    ]
    branches.append(branch("unclear", {"anyOf": [any_produce, null]}, null, null))
    branches.append(branch("unsupported", null, null, null))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "produce_assessment_strict",
            "strict": True,
            "schema": {"anyOf": branches},
        },
    }


def answer_valid(a):
    """Parsed answer check: produce/stage pairing and status consistency."""
    if a.get("status") == "assessed":
        return a.get("produce") in STAGES and a.get("ripeness") in STAGES[a["produce"]]
    if a.get("status") == "unclear":
        return a.get("ripeness") is None and a.get("produce") in (None, *PRODUCES)
    return (
        a.get("status") == "unsupported"
        and a.get("produce") is None
        and a.get("ripeness") is None
    )


def prep_image(img, short=256, long_cap=384):
    """Upscale small crops so the shorter side is at least `short` px (long side capped); larger images are kept.
    Used the same way for training, validation and the phone-form scorer."""
    from PIL import Image

    w, h = img.size
    scale = short / min(w, h)
    if scale <= 1:
        return img
    scale = min(scale, long_cap / max(w, h)) if max(w, h) * scale > long_cap else scale
    return img.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC
    )


# Banana + tomato variant (apples are not supported): VARIANT=bt
PROMPT_BT = (
    PROMPT.replace(
        "bananas, apples and tomatoes are supported",
        "bananas and tomatoes are supported",
    )
    .replace("a banana, an apple or a tomato", "a banana or a tomato")
    .replace(
        "none of these fruits (e.g. car, person, shoe, landscape)",
        "neither of these fruits (e.g. apple, car, person, shoe, landscape)",
    )
    .replace("   - Apple ripeness: 'immature', 'semi-mature', 'mature'\n", "")
    .replace(
        '"produce": "banana" | "apple" | "tomato" | null',
        '"produce": "banana" | "tomato" | null',
    )
)
if os.environ.get("VARIANT") == "bt":
    STAGES = {k: v for k, v in STAGES.items() if k != "apple"}
    PRODUCES = list(STAGES)
    PROMPT = PROMPT_BT
