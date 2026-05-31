# SIMANTAN Hybrid ML Simulation

This project contains a research-oriented simulation for the SIMANTAN architecture, focused on microwave link mismatch detection and anomaly detection using Balmon Class I Jakarta field inspection data and SIMS licensing reference data.

The main script, `test-new.py`, compares `Microwave_link.csv` against `DataSIMS.csv`, classifies detected violation types, saves a mismatch report, and then demonstrates a hybrid machine learning workflow using Isolation Forest and Random Forest. It also generates publication-quality figures for thesis or paper documentation.

## Project Files

| File | Description |
| --- | --- |
| `test-new.py` | Main research-ready simulation script. Loads data, cleans it, engineers features, trains models, evaluates performance, and saves figures. |
| `Microwave_link.csv` | Primary field inspection dataset containing microwave link records, technical parameters, compliance status, and notes. |
| `DataSIMS.csv` | SIMS licensing reference dataset used for comparison against field inspection data. |
| `SIMANTAN_Mismatch_Report.csv` | Generated mismatch report containing detected violation type, mismatch reason, field values, SIMS values, deltas, and detection flags. |
| `main.py` | Earlier prototype script for merging data and testing Isolation Forest/Random Forest logic. |
| `test.py` | Earlier experimental script with additional mismatch and feature-importance checks. |
| `Fig*.png` | Generated output figures from the main simulation. |

## Workflow Summary

The simulation performs the following steps:

1. Load microwave link field data from `Microwave_link.csv`.
2. Load SIMS reference data from `DataSIMS.csv`.
3. Normalize license IDs and auto-detect SIMS technical columns.
4. Convert SIMS bandwidth values from kHz to MHz when needed.
5. Clean frequency, bandwidth, and coordinate values.
6. Merge field records with SIMS records using `curr_lic_num`.
7. Detect mismatch flags for missing license records, TX mismatch, RX mismatch, bandwidth mismatch, TX/RX reversal, and off-air status.
8. Classify each record into a detected violation type.
9. Save `SIMANTAN_Mismatch_Report.csv`.
10. Engineer ML features such as frequency deviation, bandwidth deviation, TX/RX reversal, off-air status, license mismatch, and composite risk.
11. Apply domain-aware augmentation to expand the small real-world dataset for simulation.
12. Train an Isolation Forest model for unsupervised anomaly detection.
13. Train a Random Forest classifier for supervised validation.
14. Generate evaluation metrics and publication-ready visualizations.

## Detected Violation Types

The current mismatch logic identifies these categories:

| Violation Type | Meaning |
| --- | --- |
| `A: License Mismatch` | License number is missing from SIMS, marked as `Tidak Berizin`, or has direct TX/RX parameter mismatch that is not better explained by reversal or bandwidth deviation. |
| `B: Off Air / Warehoused` | Field record is marked `Off Air`, has `PENGGUDANGAN`, or has missing TX/RX values indicating warehoused/inactive equipment. |
| `C: TX/RX Frequency Reversed` | Field TX matches SIMS RX and field RX matches SIMS TX within tolerance. |
| `D: Bandwidth Deviation` | Field bandwidth differs from SIMS bandwidth beyond tolerance. |

## Requirements

Use Python 3.10 or newer if possible.

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## How to Run

Run the main simulation script:

```bash
python test-new.py
```

In VS Code, run the full Python file. Avoid running only selected documentation text with Code Runner, because selected bullet/comment text may be copied into `tempCodeRunnerFile.py` and executed separately.

Before running on another machine, update these paths near the top of `test-new.py`:

```python
PATH_MICROWAVE = r"C:\Users\user\Documents\UNI\Thesis\Microwave_link.csv"
PATH_SIMS = r"C:\Users\user\Documents\UNI\Thesis\DataSIMS.csv"
```

## Input Data

`Microwave_link.csv` is expected to include columns such as:

- `curr_lic_num`
- `tx_mhz`
- `rx_mhz`
- `bw_mhz`
- `koor_long`
- `koor_lat`
- `sertifikat`
- `status`
- `Keterangan`

`DataSIMS.csv` is used as the licensing reference source. The script requires a license identifier column such as `curr_lic_num`. If `DataSIMS.csv` is missing, `test-new.py` can generate synthetic SIMS records from the microwave link field data for simulation purposes.

The current SIMS file is also supported when it uses these column names:

| SIMS Column | Used As |
| --- | --- |
| `curr_lic_num` | License ID for merging with field data. |
| `freq` | SIMS TX frequency in MHz. |
| `freq_pair` | SIMS RX frequency in MHz. |
| `bwidht` | SIMS bandwidth, usually stored in kHz and converted to MHz by the script. |

## Generated Outputs

Running `test-new.py` saves this mismatch report:

| Output | Purpose |
| --- | --- |
| `SIMANTAN_Mismatch_Report.csv` | Row-level mismatch details, including detected violation type, mismatch reason, field/SIMS values, frequency and bandwidth deltas, and mismatch flags. |

It also saves these figures:

| Output | Purpose |
| --- | --- |
| `Fig1_Hybrid_ML_Performance_RealWorld.png` | Confusion matrix, ROC curve, Isolation Forest score distribution, and cross-validation F1 scores. |
| `Fig2_Feature_Importance_RealWorld.png` | Random Forest feature importance mapped to SIMANTAN dashboard priorities. |
| `Fig3_Violation_Distribution_RealWorld.png` | Distribution of detected violation categories from the Microwave vs SIMS comparison. |
| `Fig4_Geographic_Compliance_Map.png` | Geographic compliance/anomaly map based on station coordinates. |
| `Fig5_Data_Quality_Report.png` | Missing-value analysis for the microwave link dataset. |

## Notes

- The dataset is small, so `test-new.py` uses documented domain-aware augmentation to simulate a larger training set.
- Mismatch detection happens before ML training; the ML section uses the detected mismatch labels as the simulation ground truth.
- Detection tolerances are configured in `test-new.py` as `FREQ_TOLERANCE_MHZ` and `BW_TOLERANCE_MHZ`.
- The generated metrics are intended for conceptual validation of the SIMANTAN architecture, not final production model performance.
- For paper or thesis reporting, disclose whether SIMS data came from the real `DataSIMS.csv` file or from the fallback synthetic generation path.
