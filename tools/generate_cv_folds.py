"""
Generate three leave-one-driver-out cross-validation fold manifests.

Fold 1: Train = Driver E + B + D   |  Test = Driver A (5 trips)
Fold 2: Train = Driver E + A + D   |  Test = Driver B (1 trip)
Fold 3: Train = Driver E + A + B   |  Test = Driver D (1 trip)

Within each fold, 2 Driver E trips (Vf) are held out as Dev.
The remaining 60 Driver E trips are always in Train.

Output: data/folds/fold-{1,2,3}.json  (schema navdr.split.v1)
        data/folds/model_card.md
"""
import hashlib, json, sys
from pathlib import Path

DRIVER_MAP = {
    "S (Driver A)":   "A",
    "M (Driver B)":   "B",
    "Y (Driver D)":   "D",
    "Vf (Driver E)":  "E",
    "Vta (Driver E)": "E",
    "Vtb (Driver E)": "E",
    "Vw (Driver E)":  "E",
}

DEV_VEHICLE_CODE = "Vf"  # 2 trips, always held out as dev

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    inv_path = Path("data/proposed_inventory.json")
    if not inv_path.is_file():
        inv_path = Path(__file__).resolve().parents[1] / "data/proposed_inventory.json"
    inv = json.loads(inv_path.read_text())
    approved = [t for t in inv["trips"] if t["approved"]]

    # Tag each trip with its short driver label
    for t in approved:
        t["_driver"] = DRIVER_MAP[t["driver"]]
        t["_is_dev_e"] = t["id"].startswith("V-Vf")  # Vf trips → dev

    drivers_minority = {"A", "B", "D"}

    folds = [
        {"name": "fold-1", "test_driver": "A", "train_drivers": {"B", "D"}},
        {"name": "fold-2", "test_driver": "B", "train_drivers": {"A", "D"}},
        {"name": "fold-3", "test_driver": "D", "train_drivers": {"A", "B"}},
    ]

    out_dir = Path("data/folds")
    if not out_dir.parent.is_dir():
        out_dir = Path(__file__).resolve().parents[1] / "data/folds"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    repo_root = inv_path.parent.parent

    for fold in folds:
        trips_out = []
        counts = {"train": 0, "dev": 0, "test": 0}

        for t in approved:
            rel_path = f"data/canonical/{t['id']}.csv"
            p = repo_root / rel_path
            if not p.is_file():
                p = Path(rel_path)
            sha = digest(p)
            entry = {
                "id":      t["id"],
                "path":    rel_path,
                "sha256":  sha,
                "vehicle": t["driver"],
                "route":   t["route"],
            }

            if t["_driver"] == fold["test_driver"]:
                entry["split"] = "test"
            elif t["_driver"] == "E" and t["_is_dev_e"]:
                entry["split"] = "dev"
            else:
                # Driver E (non-dev) + the two non-test minority drivers
                entry["split"] = "train"

            entry["group"] = entry["id"]
            counts[entry["split"]] += 1
            trips_out.append(entry)

        manifest = {
            "schema":    "navdr.split.v1",
            "status":    "DRAFT_REQUIRES_USER_REVIEW",
            "synthetic": False,
            "seed":      26168,
            "grouping":  "leave-one-driver-out cross-validation",
            "fold":      fold["name"],
            "testDriver": fold["test_driver"],
            "trainMinorityDrivers": sorted(fold["train_drivers"]),
            "warnings":  [
                "Driver E (90% of data) is present in every fold's training set.",
                "Phone field omitted; all trips share the same device.",
            ],
            "trips": trips_out,
        }

        fold_path = out_dir / f"{fold['name']}.json"
        fold_path.write_text(json.dumps(manifest, indent=2))
        fold_sha = digest(fold_path)

        summary.append({
            "fold":        fold["name"],
            "test_driver": fold["test_driver"],
            "train":       counts["train"],
            "dev":         counts["dev"],
            "test":        counts["test"],
            "sha256":      fold_sha,
            "path":        fold_path.as_posix(),
        })

        print(f"{fold['name']}: test=Driver {fold['test_driver']} "
              f"({counts['test']} trips)  train={counts['train']}  "
              f"dev={counts['dev']}  SHA256={fold_sha[:16]}...")

    # ── Model card ──────────────────────────────────────────────────
    card = f"""# NavDR TCN Model Card — Leave-One-Driver-Out Cross-Validation

## Evaluation Design

Three-fold cross-validation, each fold training on Driver E (62 trips across
four vehicles) plus two of three minority drivers, testing on the remaining
minority driver.

| Fold | Train drivers | Dev | Test driver | Test trips |
|------|--------------|-----|-------------|------------|
| fold-1 | E + B + D | E (Vf, 2 trips) | **A** | {summary[0]['test']} |
| fold-2 | E + A + D | E (Vf, 2 trips) | **B** | {summary[1]['test']} |
| fold-3 | E + A + B | E (Vf, 2 trips) | **D** | {summary[2]['test']} |

## Manifest SHA-256 Hashes

| Fold | SHA-256 |
|------|---------|
| fold-1 | `{summary[0]['sha256']}` |
| fold-2 | `{summary[1]['sha256']}` |
| fold-3 | `{summary[2]['sha256']}` |

## What This Evaluation Measures

Each fold tests whether the TCN can predict forward velocity for a driver
whose personal driving style was **never seen during training**. The model
has seen the two other minority drivers and all of Driver E's data.

Results for each fold must be reported **independently**, not averaged into
a single number. Averaging three folds where two have N=1 test trips would
produce a misleading aggregate.

## Known Limitations — Read Before Interpreting Results

1. **Driver E is in every training fold.** This is unavoidable: Driver E
   contributed 62 of 69 approved trips (90%). The model has always seen
   Driver E's style during training. We cannot evaluate generalisation
   away from Driver E with this dataset.

2. **Small-sample test sets.** Fold 2 tests on 1 trip (Driver B) and
   Fold 3 tests on 1 trip (Driver D). A single bad trip can swing the
   error metric dramatically. These results are **preliminary evidence,
   not confident accuracy claims**.

3. **Same phone, same city.** All trips were collected with the same
   smartphone model in the same metropolitan area. Performance on a
   different phone, vehicle type, or geography is unknown.

## Required Next Steps Before Treating This as Final

> **The team must prioritise collecting additional real trips from more
> drivers before this evaluation is treated as a final accuracy result.**

Specifically:
- At least 3 additional drivers, each with ≥5 trips, would allow a
  meaningful leave-one-driver-out design with statistical power.
- Trips from a different city or phone model would test geographic and
  hardware generalisation.
- Until then, report all results with the caveat: *"Preliminary;
  N=1 test folds and single-phone/single-city data."*

## Data Provenance

- Source: IO-VNBD Synchronised Categorised Dataset (Onyekpe et al.)
- Preprocessing: `tools/prepare_io_vnbd.py` — gravity correction,
  unit conversion, 10 Hz resampling
- 69 approved trips, 2 excluded (Vw01/Vw15: stationary), 1 skipped
  (S3b: bad clock overlap)
- Canonical format: `timestampNs, ax, ay, az, gx, gy, gz, speedMps`
- Ground-truth speed: vehicle VBOX GPS (10 Hz)
"""

    card_path = out_dir / "model_card.md"
    card_path.write_text(card)
    print(f"\nModel card: {card_path}")
    print("Fold manifests ready for user review.")

if __name__ == "__main__":
    main()
