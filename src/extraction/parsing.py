import json
import re
from typing import TypeVar

from pydantic import BaseModel

from ..debug import dbg

T = TypeVar("T", bound=BaseModel)

_ETAPA = "extraction.parsing"


class ExtractionError(Exception):
    pass


def parse_llm_json(raw: str, model: type[T]) -> T:
    cleaned = _strip_markdown(raw)
    if cleaned != raw.strip():
        dbg(_ETAPA, "cercas de markdown removidas da resposta")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"LLM returned invalid JSON: {e}\nRaw output: {raw[:300]}") from e
    try:
        validado = model.model_validate(data)
    except Exception as e:
        raise ExtractionError(f"Schema validation failed: {e}") from e
    dbg(_ETAPA, f"JSON validado contra {model.__name__}")
    return validado


def _strip_markdown(text: str) -> str:
    """Remove ```json ... ``` fences that some models add despite instructions."""
    cleaned = text.strip()
    match = re.match(r"^```(?:json)?\s*([\s\S]*?)```$", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned
