# NavDR TCN Model Card — Leave-One-Driver-Out Cross-Validation

## Evaluation Design

Three-fold cross-validation, each fold training on Driver E (62 trips across
four vehicles) plus two of three minority drivers, testing on the remaining
minority driver.

| Fold | Train drivers | Dev | Test driver | Test trips |
|------|--------------|-----|-------------|------------|
| fold-1 | E + B + D | E (Vf, 2 trips) | **A** | 5 |
| fold-2 | E + A + D | E (Vf, 2 trips) | **B** | 1 |
| fold-3 | E + A + B | E (Vf, 2 trips) | **D** | 1 |

## Manifest SHA-256 Hashes

| Fold | SHA-256 |
|------|---------|
| fold-1 | `f1fbf7fc26621874a893d6cb0b5c19c8bfefd41dd06be3495c28ce7bf080f86f` |
| fold-2 | `a85316ff2926e2c9e1123f2b542e7946feb6e25049bca49943e36b78a6fc6e83` |
| fold-3 | `f6dede1057b981e090a07d5e696fdd45ba30742dac8b4c4ca855f2c97838feed` |

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
