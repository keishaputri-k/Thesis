# Microwave Link Compliance and Anomaly Detection System

This project is a research-oriented decision-support system for checking microwave link field inspection records against licensing reference data. It identifies technical and administrative mismatches between field inspection records and reference licensing records, then uses machine learning to simulate automated anomaly detection from those mismatch patterns.

The active script is `main.py`. It loads field and reference CSV files, detects violation categories, builds a row-level mismatch report, trains anomaly-detection and classification models, and generates figures for thesis or paper documentation.

## System Purpose

The system supports spectrum monitoring and inspection analysis by:

- Matching field inspection records to licensing records with `curr_lic_num`.
- Comparing field TX frequency, RX frequency, and bandwidth against licensed parameters.
- Detecting unmatched or missing licensing records.
- Detecting off-air or warehoused equipment.
- Detecting reversed TX/RX frequency pairs.
- Detecting bandwidth deviations.
- Creating a structured mismatch report for review.
- Training machine learning models on engineered compliance and mismatch features.

## Project Files

| File | Description |
| --- | --- |
| `main.py` | Main script for data loading, mismatch detection, feature engineering, machine learning, evaluation, and figure generation. |
| `Microwave_link.csv` | Field inspection dataset containing microwave link records, technical parameters, status, coordinates, and inspection notes. |
| `DataSIMS.csv` | Licensing reference dataset used to compare against the field inspection records. |
| `Fig*.png` | Generated figures from the model evaluation and data analysis workflow. |
| `README.md` | Project explanation and running instructions. |

## Detected Violation Types

| Violation Type | Meaning |
| --- | --- |
| `A: License Mismatch` | License number is missing from the reference data, marked as `Tidak Berizin`, or has a direct TX/RX mismatch not better explained by reversal or bandwidth deviation. |
| `B: Off Air / Warehoused` | Field record is marked `Off Air`, has `PENGGUDANGAN`, or has missing TX/RX values that indicate inactive equipment. |
| `C: TX/RX Frequency Reversed` | Field TX matches reference RX and field RX matches reference TX within tolerance. |
| `D: Bandwidth Deviation` | Field bandwidth differs from reference bandwidth beyond tolerance. |

## Data Processing Workflow

1. Load field records from `Microwave_link.csv`.
2. Load reference records from `DataSIMS.csv`.
3. Normalize license IDs for reliable matching.
4. Detect reference columns such as `freq`, `freq_pair`, and `bwidht`.
5. Convert reference bandwidth from kHz to MHz when needed.
6. Clean frequency, bandwidth, and coordinate values.
7. Merge field and reference records with `curr_lic_num`.
8. Calculate TX, RX, and bandwidth deltas.
9. Create mismatch flags for license mismatch, TX mismatch, RX mismatch, bandwidth mismatch, TX/RX reversal, and off-air status.
10. Assign each record to a detected violation type.
11. Save a mismatch report CSV using the configured output path in `main.py`.

## Machine Learning Pipeline

1. **Feature Engineering**
   The script builds numerical features from field and reference data, including cleaned TX/RX frequency, cleaned bandwidth, certification flag, coordinates, TX/RX/bandwidth deltas, mismatch flags, and a composite risk score.

2. **Label Construction**
   Detected violation types are converted into binary labels. `0` means compliant, and `1` means anomaly or violation.

3. **Data Augmentation**
   The original field dataset is small, so the script expands it to a configured target size using controlled noise around real records. This is used for simulation and should be disclosed in reporting.

4. **Feature Scaling**
   The engineered numerical features are standardized with `StandardScaler`.

5. **Isolation Forest**
   Isolation Forest is used as the unsupervised anomaly detection stage. It learns the structure of the engineered feature space and flags unusual records.

6. **Random Forest Classifier**
   Random Forest is used as the supervised validation stage. It learns from the compliant/anomaly labels and produces classification metrics.

7. **Evaluation**
   The script reports accuracy, F1 score, ROC-AUC, a classification report, confusion matrix, and 5-fold cross-validation F1 score.

8. **Visualization**
   The script generates figures for model performance, feature importance, violation distribution, geographic compliance mapping, and data quality.

## Requirements

Use Python 3.10 or newer if possible.

Install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## How to Run

Run the main script:

```bash
python main.py
```

In VS Code, run the full Python file. Avoid running only selected documentation text with Code Runner, because selected text may be copied into `tempCodeRunnerFile.py` and executed separately.

Before running on another machine, update these paths near the top of `main.py`:

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

`DataSIMS.csv` is expected to include a license identifier column such as `curr_lic_num`.

The current reference file is supported when it uses these columns:

| Reference Column | Used As |
| --- | --- |
| `curr_lic_num` | License ID for matching field records. |
| `freq` | Reference TX frequency in MHz. |
| `freq_pair` | Reference RX frequency in MHz. |
| `bwidht` | Reference bandwidth, usually stored in kHz and converted to MHz by the script. |

## Generated Outputs

Running `main.py` generates a mismatch report CSV with row-level details such as detected violation type, mismatch reason, field values, reference values, parameter deltas, and mismatch flags.

It also saves these figures:

| Output | Purpose |
| --- | --- |
| `Fig1_Hybrid_ML_Performance_RealWorld.png` | Confusion matrix, ROC curve, Isolation Forest score distribution, and cross-validation F1 scores. |
| `Fig2_Feature_Importance_RealWorld.png` | Random Forest feature importance for engineered inspection and mismatch features. |
| `Fig3_Violation_Distribution_RealWorld.png` | Distribution of detected violation categories from the field vs reference comparison. |
| `Fig4_Geographic_Compliance_Map.png` | Geographic compliance/anomaly map based on station coordinates. |
| `Fig5_Data_Quality_Report.png` | Missing-value analysis for the field inspection dataset. |

## Notes

- Mismatch detection is performed before machine learning training.
- The machine learning models use detected mismatch labels as simulation ground truth.
- Detection tolerances are configured in `main.py` as `FREQ_TOLERANCE_MHZ` and `BW_TOLERANCE_MHZ`.
- The generated metrics are intended for conceptual validation, not final production model performance.
- If the reference file is missing, the script can generate synthetic reference records from field data for simulation purposes.
