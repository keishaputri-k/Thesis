# Microwave Link Compliance and Anomaly Detection System

This project is a research-oriented decision-support system for checking microwave link field inspection records against licensing reference data using the Isolation forest and random forest method. It helps identify technical and administrative mismatches between real inspection data and the SIMS reference database, then uses machine learning to simulate how those mismatch patterns could support automated anomaly detection.

The main script, `test-new.py`, compares `Microwave_link.csv` with `DataSIMS.csv`, classifies violation types, saves a row-level mismatch report, and generates evaluation figures for thesis or paper documentation.

## What the System Does

The system supports spectrum monitoring and field inspection analysis by:

- Matching microwave link inspection records with licensing records using `curr_lic_num`.
- Checking whether field TX frequency, RX frequency, and bandwidth match the licensed parameters.
- Detecting missing or unmatched license records.
- Detecting equipment that is off air or warehoused.
- Detecting reversed TX/RX frequency pairs.
- Detecting bandwidth deviation.
- Producing a structured mismatch report for review.
- Training machine learning models to simulate automated anomaly detection from engineered inspection features.

## Project Files

| File | Description |
| --- | --- |
| `test-new.py` | Main research-ready script for data loading, mismatch detection, feature engineering, model training, evaluation, and figure generation. |
| `Microwave_link.csv` | Primary field inspection dataset containing microwave link records, technical parameters, compliance status, and inspection notes. |
| `DataSIMS.csv` | Licensing reference dataset used to compare against field inspection data. |
| Mismatch report CSV | Generated row-level report containing detected violation type, mismatch reason, field values, reference values, deltas, and detection flags. The output name is controlled by `OUTPUT_MISMATCH_REPORT` in `test-new.py`. |
| `main.py` | Earlier prototype script for merge and model testing. |
| `test.py` | Earlier experimental script with additional mismatch and feature-importance checks. |
| `Fig*.png` | Generated output figures from the main simulation. |

## Detected Violation Types

| Violation Type | Meaning |
| --- | --- |
| `A: License Mismatch` | License number is missing from the reference data, marked as `Tidak Berizin`, or has direct TX/RX parameter mismatch that is not better explained by reversal or bandwidth deviation. |
| `B: Off Air / Warehoused` | Field record is marked `Off Air`, has `PENGGUDANGAN`, or has missing TX/RX values indicating inactive or warehoused equipment. |
| `C: TX/RX Frequency Reversed` | Field TX matches reference RX and field RX matches reference TX within tolerance. |
| `D: Bandwidth Deviation` | Field bandwidth differs from reference bandwidth beyond tolerance. |

## Data Processing Workflow

1. Load microwave link field records from `Microwave_link.csv`.
2. Load licensing reference records from `DataSIMS.csv`.
3. Normalize license IDs so records can be matched reliably.
4. Auto-detect reference columns such as `freq`, `freq_pair`, and `bwidht`.
5. Convert reference bandwidth from kHz to MHz when needed.
6. Clean frequency, bandwidth, and coordinate values.
7. Merge field records with reference records using `curr_lic_num`.
8. Calculate TX, RX, and bandwidth deltas.
9. Create mismatch flags for license mismatch, TX mismatch, RX mismatch, bandwidth mismatch, TX/RX reversal, and off-air status.
10. Classify each record into a detected violation type.
11. Save the mismatch report CSV.

## Machine Learning Pipeline

The machine learning section is used as a conceptual simulation layer after deterministic mismatch detection has produced labels and engineered features.

1. **Feature Engineering**
   The script builds numerical features from inspection and reference data, including cleaned TX/RX frequency, cleaned bandwidth, certification flag, coordinates, TX/RX/bandwidth deltas, mismatch flags, and a composite risk score.

2. **Ground-Truth Label Construction**
   Detected violation types are converted into binary labels:
   `0` means compliant and `1` means anomaly or violation.

3. **Domain-Aware Data Augmentation**
   Because the real field dataset is small, the script expands the dataset to a configured target size using controlled noise around real records. This is used only for simulation and should be disclosed when reporting results.

4. **Feature Scaling**
   Numerical features are standardized with `StandardScaler` so model training is less affected by differences in feature scale.

5. **Isolation Forest**
   Isolation Forest is used as an unsupervised anomaly detection model. It learns the distribution of the engineered feature space and flags records that appear unusual.

6. **Random Forest Classifier**
   Random Forest is used as a supervised validation model. It learns from the generated compliant/anomaly labels and produces classification metrics.

7. **Evaluation**
   The script reports accuracy, F1 score, ROC-AUC, classification report, confusion matrix, and 5-fold cross-validation F1 score.

8. **Visualization**
   The script generates figures for model performance, feature importance, violation distribution, geographic compliance mapping, and data quality.

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

In VS Code, run the full Python file. Avoid running only selected documentation text with Code Runner, because selected text may be copied into `tempCodeRunnerFile.py` and executed separately.

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

`DataSIMS.csv` is used as the licensing reference source. The script requires a license identifier column such as `curr_lic_num`.

The current reference file is also supported when it uses these column names:

| Reference Column | Used As |
| --- | --- |
| `curr_lic_num` | License ID for merging with field data. |
| `freq` | Reference TX frequency in MHz. |
| `freq_pair` | Reference RX frequency in MHz. |
| `bwidht` | Reference bandwidth, usually stored in kHz and converted to MHz by the script. |

## Generated Outputs

Running `test-new.py` saves a mismatch report CSV with row-level mismatch details.

It also saves these figures:

| Output | Purpose |
| --- | --- |
| `Fig1_Hybrid_ML_Performance_RealWorld.png` | Confusion matrix, ROC curve, Isolation Forest score distribution, and cross-validation F1 scores. |
| `Fig2_Feature_Importance_RealWorld.png` | Random Forest feature importance for the engineered mismatch and inspection features. |
| `Fig3_Violation_Distribution_RealWorld.png` | Distribution of detected violation categories from the field vs reference comparison. |
| `Fig4_Geographic_Compliance_Map.png` | Geographic compliance/anomaly map based on station coordinates. |
| `Fig5_Data_Quality_Report.png` | Missing-value analysis for the microwave link dataset. |

## Notes

- Mismatch detection happens before machine learning training.
- The machine learning models use detected mismatch labels as simulation ground truth.
- Detection tolerances are configured in `test-new.py` as `FREQ_TOLERANCE_MHZ` and `BW_TOLERANCE_MHZ`.
- The generated metrics are intended for conceptual validation, not final production model performance.
- If the reference file is missing, the script can generate synthetic reference records from field data for simulation purposes.
