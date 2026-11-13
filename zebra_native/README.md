# zebra_native

Goal: turn a pickled ZeBRA / zcor model into a native binary that reads patient
JSON and prints risk scores, with no Python at run time.

```
MODEL.pkl.gz --(export_model.py, once, needs Python + zcor)--> export dir
export dir   --(build)--> zebra_score binary (+ model data)
patients.json --(zebra_score)--> scores
```

## Status

| Stage | What | State |
|---|---|---|
| 1 | Export the LightGBM models and evaluate them natively | Done, bit-exact against Python |
| 2 | Port feature building (JSON → model inputs) to C++ | Not started |
| 3 | `zebra_score` binary: JSON in, scores out; embed model data | Not started |

Stage 1 results for `ILD-AFSTDPULMO_1-0-0` (zcor 1.1.10): every one of the 8
models (7 base + FINAL) matches Python with a difference of exactly 0, on the
5 sample patients (end to end from the captured model inputs) and on 2,000
random stress rows per model (missing values, zeros, values on split
thresholds).

## Layout

- `export/export_model.py` – writes `manifest.json` (model order, input column
  order, config) and `trees/<NAME>.txt` (LightGBM text dumps). With
  `--reference PATIENTS.json` it also runs Python's `predict_json` and records
  the exact input matrix and output of every LightGBM call in `reference/`.
- `src/gbdt.hpp` – dependency-free LightGBM evaluator (numerical splits,
  binary objective).
- `tools/check_trees.cpp` – compares native and Python scores on `reference/`
  or `fuzz/`.
- `tools/make_tree_fuzz.py` – writes random stress inputs scored by Python.
- `third_party/nlohmann/json.hpp` – JSON for Modern C++ v3.11.3 (MIT).

## Running stage 1

Needs the zcor version the model was built with (1.1.10 for the ILD model),
on Python 3.10.

```bash
python export/export_model.py MODEL.pkl.gz OUT --reference PATIENTS.json
python tools/make_tree_fuzz.py MODEL.pkl.gz OUT
cmake -S . -B build && cmake --build build
build/check_trees OUT
build/check_trees OUT fuzz
```

## Notes

- **Sigmoid precision.** LightGBM's text format prints the binary `sigmoid`
  parameter with 6 significant digits (e.g. `85.8471`, trained as
  `85.8471198466548`). A pickled LightGBM model is rebuilt from that text, so
  the Python model in use runs with the rounded value. The native evaluator
  reads it from the tree file and therefore matches; using the training value
  instead gives differences of about 1e-7.
- Trees are summed in file order and passed through
  `1 / (1 + exp(-sigmoid * raw))`, as LightGBM does. Do not build with
  `-ffast-math`.
- Base-model inputs are the zcor column lists minus `patient_id`, followed by
  `age_at_screening` and `sex` (`M` → 1, anything else → 0). For `SEQ_*` models
  `_code_density` is renamed to `_density`. FINAL takes the 7 base scores in
  `BASE_MODEL_NAMES` order, then age and sex.
- `predict_json` sets the risk to 1 for patients whose window already contains
  a target code (`MARK_OBVIOUS_POSITIVES`, on by default). Stage 3 must do the same.
