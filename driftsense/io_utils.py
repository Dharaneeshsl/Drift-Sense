"""Input validation, pair discovery, and output serialization for DriftSense-FM."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

OFFICIAL_SHAPE = (1000, 1000)


def load_array(path: str | Path, name: str, official: bool = True) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} file does not exist: {p}")
    try:
        arr = np.load(p, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Could not load {name} array {p}: {exc}") from exc
    if not isinstance(arr, np.ndarray) or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be a numeric NumPy array: {p}")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D or (H,W,1); received shape {arr.shape}")
    if official and arr.shape != OFFICIAL_SHAPE:
        raise ValueError(f"{name} must have shape {OFFICIAL_SHAPE} in official mode; received {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(arr, dtype=np.float32)


def discover_pairs(input_dir: str | Path, official: bool = True) -> list[dict[str, str]]:
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    pairs_csv = root / "pairs.csv"
    pairs: list[dict[str, str]] = []
    if pairs_csv.exists():
        with pairs_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            required = {"id", "reference", "search"}
            if not required.issubset(fieldnames):
                raise ValueError("pairs.csv must contain columns id,reference,search")
            rows = list(reader)
        if not rows:
            raise ValueError("pairs.csv contains no input pairs")
        seen: set[str] = set()
        for index, row in enumerate(rows, start=2):
            ident = (row.get("id") or "").strip()
            reference_value = (row.get("reference") or "").strip()
            search_value = (row.get("search") or "").strip()
            if not ident or not reference_value or not search_value:
                raise ValueError(f"pairs.csv row {index} has an empty id, reference, or search field")
            if ident in seen:
                raise ValueError(f"pairs.csv contains duplicate id: {ident}")
            seen.add(ident)
            reference_path = Path(reference_value)
            search_path = Path(search_value)
            if not reference_path.is_absolute():
                reference_path = root / reference_path
            if not search_path.is_absolute():
                search_path = root / search_path
            if not reference_path.is_file():
                raise FileNotFoundError(f"pairs.csv row {index} reference file does not exist: {reference_path}")
            if not search_path.is_file():
                raise FileNotFoundError(f"pairs.csv row {index} search file does not exist: {search_path}")
            pairs.append({"id": ident, "reference": str(reference_path.resolve()), "search": str(search_path.resolve())})
        return pairs
    ref = root / "reference.npy"
    search = root / "search.npy"
    if ref.exists() and search.exists():
        return [{"id": "pair", "reference": str(ref.resolve()), "search": str(search.resolve())}]
    if ref.exists() != search.exists():
        raise ValueError("Input directory must contain both reference.npy and search.npy when using the single-pair form")
    for ref in sorted(root.glob("*_reference.npy")):
        ident = ref.name[: -len("_reference.npy")]
        candidate = root / f"{ident}_search.npy"
        if candidate.exists():
            pairs.append({"id": ident, "reference": str(ref), "search": str(candidate)})
    if len({row["id"] for row in pairs}) != len(pairs):
        raise ValueError("Input pair IDs must be unique")
    if not pairs:
        raise ValueError("No input pair found. Provide pairs.csv, reference.npy+search.npy, or <id>_reference.npy/<id>_search.npy pairs.")
    return pairs


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_outputs(output_dir: str | Path, predictions: list[dict[str, Any]], diagnostics: dict[str, dict[str, Any]]) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x", "y", "confidence"])
        writer.writeheader()
        for row in predictions:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    for ident, diag in diagnostics.items():
        (root / f"{ident}.json").write_text(json.dumps(json_safe(diag), indent=2), encoding="utf-8")
