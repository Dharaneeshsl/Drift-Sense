# Contributing to DriftSense-FM

## Development setup

Use Python 3.11 and install the pinned runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

## Required verification

Before opening a pull request, run:

```bash
python -m unittest discover -s tests -v
python submission_smoke.py
python run.py --help
```

## Scope rules

- Keep the baseline deterministic and offline.
- Do not add network calls, API keys, or implicit model downloads.
- Preserve the official 1000x1000 input contract unless the interface is versioned.
- Add regression tests for behavior changes.
- Keep benchmark claims clearly separated from hidden or official challenge results.
