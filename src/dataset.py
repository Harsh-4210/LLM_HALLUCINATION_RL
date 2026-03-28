import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_LABELS = {
    "correct_cautious",
    "correct_confident",
    "wrong_cautious",
    "wrong_confident",
}


@dataclass
class Sample:
    id: str
    domain: str
    response: str
    label: str
    confidence_markers: list[str]
    metadata: dict[str, Any]

    @property
    def is_risky(self) -> int:
        return 1 if self.label == "wrong_confident" else 0


def _validate_sample(raw: dict[str, Any]) -> Sample:
    required = ["id", "domain", "response", "label"]
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(f"Missing required fields {missing} in sample: {raw}")

    if raw["label"] not in ALLOWED_LABELS:
        raise ValueError(f"Invalid label '{raw['label']}' for sample id={raw['id']}")

    return Sample(
        id=str(raw["id"]),
        domain=str(raw["domain"]),
        response=str(raw["response"]),
        label=str(raw["label"]),
        confidence_markers=list(raw.get("confidence_markers", [])),
        metadata=dict(raw.get("metadata", {})),
    )


def load_dataset(path: str | Path) -> list[Sample]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    samples: list[Sample] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            samples.append(_validate_sample(raw))

    if not samples:
        raise ValueError("Dataset is empty")

    return samples
