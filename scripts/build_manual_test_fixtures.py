from __future__ import annotations

import argparse
import importlib
import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path
from uuid import uuid4
from xml.sax.saxutils import escape

import joblib
import onnx
import pandas as pd
import torch
from onnx import TensorProto, helper
from sklearn.ensemble import RandomForestClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_SOURCE_DIR = REPO_ROOT / "api" / "tests" / "fixtures" / "datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "api" / "tests" / "fixtures" / "manual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manual signoff fixtures for Painel Amanaje.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _column_letter(index: int) -> str:
    result = ""
    current = index
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_minimal_xlsx(dataframe: pd.DataFrame, destination: Path) -> None:
    rows = [list(dataframe.columns)] + dataframe.astype(object).where(pd.notna(dataframe), "").values.tolist()
    shared_strings: list[str] = []
    shared_string_index: dict[str, int] = {}
    sheet_rows: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_index, value in enumerate(row, start=1):
            cell_ref = f"{_column_letter(column_index)}{row_index}"
            if value in ("", None):
                cells.append(f'<c r="{cell_ref}"/>')
                continue
            if isinstance(value, bool):
                numeric_value = "1" if value else "0"
                cells.append(f'<c r="{cell_ref}" t="b"><v>{numeric_value}</v></c>')
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
                continue

            text_value = str(value)
            string_id = shared_string_index.get(text_value)
            if string_id is None:
                string_id = len(shared_strings)
                shared_string_index[text_value] = string_id
                shared_strings.append(text_value)
            cells.append(f'<c r="{cell_ref}" t="s"><v>{string_id}</v></c>')

        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    shared_strings_xml = "".join(f"<si><t>{escape(value)}</t></si>" for value in shared_strings)
    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
        "</Relationships>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        "</styleSheet>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    shared_strings_document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">'
        f"{shared_strings_xml}</sst>"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
        archive.writestr("xl/sharedStrings.xml", shared_strings_document)
        archive.writestr("xl/styles.xml", styles_xml)


def build_dataset_fixtures(output_dir: Path) -> dict[str, str]:
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    narrow_csv = DATASET_SOURCE_DIR / "narrow_sample.csv"
    dataframe = pd.read_csv(narrow_csv)

    xlsx_path = datasets_dir / "narrow_sample.xlsx"
    parquet_path = datasets_dir / "narrow_sample.parquet"
    write_minimal_xlsx(dataframe, xlsx_path)
    dataframe.to_parquet(parquet_path, index=False)

    return {
        "narrow_sample_xlsx": str(xlsx_path.relative_to(REPO_ROOT)),
        "narrow_sample_parquet": str(parquet_path.relative_to(REPO_ROOT)),
    }


def build_model_fixture_seed() -> tuple[list[list[float]], list[int]]:
    features = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]]
    targets = [0, 1, 1, 0, 1, 1]
    return features, targets


def write_missing_module_pickle(destination: Path) -> None:
    temp_root = destination.parent / f".tmp_missing_module_{uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    module_name = "_painel_missing_module_fixture"
    module_file = temp_root / f"{module_name}.py"
    module_file.write_text(
        "class MissingDependencyModel:\n"
        "    def predict(self, rows):\n"
        "        return [0 for _ in range(len(rows))]\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(temp_root))
    try:
        module = importlib.import_module(module_name)
        with destination.open("wb") as handle:
            pickle.dump(module.MissingDependencyModel(), handle)
    finally:
        sys.path = [entry for entry in sys.path if entry != str(temp_root)]
        sys.modules.pop(module_name, None)
        shutil.rmtree(temp_root, ignore_errors=True)


def build_onnx_identity_model(destination: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", inputs=["input"], outputs=["output"])],
        "manual_identity_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
    )
    model = helper.make_model(graph, producer_name="painel-manual-fixture")
    onnx.save(model, destination)


def build_model_fixtures(output_dir: Path) -> dict[str, str]:
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    features, targets = build_model_fixture_seed()
    estimator = RandomForestClassifier(n_estimators=8, random_state=42)
    estimator.fit(features, targets)

    joblib_path = models_dir / "valid_sklearn.joblib"
    pickle_path = models_dir / "valid_importable.pkl"
    broken_pickle_path = models_dir / "broken_missing_module.pkl"
    torchscript_path = models_dir / "valid_torchscript.pt"
    state_dict_path = models_dir / "invalid_state_dict.pt"
    onnx_path = models_dir / "identity.onnx"

    joblib.dump(estimator, joblib_path)
    with pickle_path.open("wb") as handle:
        pickle.dump(estimator, handle)
    write_missing_module_pickle(broken_pickle_path)

    class TinyScriptModel(torch.nn.Module):
        def forward(self, x):  # type: ignore[override]
            return x[:, :1]

    scripted_model = torch.jit.trace(TinyScriptModel(), torch.tensor([[0.0, 1.0]], dtype=torch.float32))
    torch.jit.save(scripted_model, str(torchscript_path))
    torch.save(TinyScriptModel().state_dict(), str(state_dict_path))
    build_onnx_identity_model(onnx_path)

    return {
        "valid_joblib": str(joblib_path.relative_to(REPO_ROOT)),
        "valid_pickle": str(pickle_path.relative_to(REPO_ROOT)),
        "broken_missing_module_pickle": str(broken_pickle_path.relative_to(REPO_ROOT)),
        "valid_torchscript": str(torchscript_path.relative_to(REPO_ROOT)),
        "invalid_state_dict": str(state_dict_path.relative_to(REPO_ROOT)),
        "identity_onnx": str(onnx_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = build_dataset_fixtures(output_dir)
    model_manifest = build_model_fixtures(output_dir)

    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "datasets": dataset_manifest,
        "models": model_manifest,
        "source_fixtures": {
            "narrow_sample_csv": str((DATASET_SOURCE_DIR / "narrow_sample.csv").relative_to(REPO_ROOT)),
            "wide_columns_csv": str((DATASET_SOURCE_DIR / "wide_columns.csv").relative_to(REPO_ROOT)),
            "github_vs_git_schema_mismatch_csv": str((DATASET_SOURCE_DIR / "github_vs_git_schema_mismatch.csv").relative_to(REPO_ROOT)),
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path.relative_to(REPO_ROOT))}, indent=2))


if __name__ == "__main__":
    main()
