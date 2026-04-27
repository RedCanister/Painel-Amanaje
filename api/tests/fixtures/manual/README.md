# Manual Fixture Package

This folder holds the **generated manual signoff fixtures** used by the repository's human QA checklist.

Generate or refresh the package with:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_manual_test_fixtures.py
```

Expected outputs:

- `datasets/narrow_sample.xlsx`
- `datasets/narrow_sample.parquet`
- `models/valid_sklearn.joblib`
- `models/valid_importable.pkl`
- `models/broken_missing_module.pkl`
- `models/valid_torchscript.pt`
- `models/invalid_state_dict.pt`
- `models/identity.onnx`
- `manifest.json`

The source datasets used to build these files live in:

- `api/tests/fixtures/datasets/`

The manual signoff instructions live in:

- `docs/manual-signoff-checklist.md`
- `docs/manual-signoff-report-template.md`
