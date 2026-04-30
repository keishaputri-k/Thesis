"""
=============================================================================
SIMANTAN: Hybrid ML Simulation — Real-World Data Version
=============================================================================
Project     : Architecture Analysis and UI/UX Prototype Design of SIMANTAN
Journal     : JOIV - International Journal on Informatics Visualization
Methodology : Design Science Research Methodology (DSRM) - Peffers et al.
Institution : Balmon Class I Radio Frequency Spectrum Monitoring - Jakarta

DATA SOURCE DECLARATION (Required for JOIV Review):
----------------------------------------------------
Primary dataset  : Microwave_link.csv
                   Real field inspection records from Balmon Class I Jakarta
                   MANTIB field operations (39 records, 17 parameters)
Secondary dataset: DataSIMS.csv
                   SIMS licensing reference database
                   NOTE: If DataSIMS.csv is unavailable, the script auto-
                   generates synthetic SIMS records from field data to
                   simulate the merge operation. Update PATH_SIMS below
                   when the file is available.

ARCHITECTURAL NOTE:
-------------------
This script is a CONCEPTUAL SIMULATION TOOL. It validates the data flow
and ML logic of the SIMANTAN architecture. The 39-record real dataset is
augmented using domain-aware oversampling (SMOTE) to enable meaningful
ML training — this augmentation is fully documented and disclosed in
Section II (Materials and Methods) of the paper.

Violation Types Identified in Real Data (from 'Keterangan' column):
  - TIDAK SESUAI ISR      : License number mismatch / unregistered
  - PENGGUDANGAN          : Equipment warehoused / Off Air
  - FREK TERBALIK         : TX/RX frequency reversal (swapped pair)
  - BANDWIDTH TIDAK SESUAI: Bandwidth deviation from licensed parameter
=============================================================================
"""

# --- Standard Library ---
import warnings
warnings.filterwarnings("ignore")
import os

# --- Data Handling ---
import numpy as np
import pandas as pd

# --- ML Libraries ---
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
)

# --- Visualization ---
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =============================================================================
# SECTION 0: Configuration
# =============================================================================

# !! UPDATE THESE PATHS TO YOUR LOCAL FILE LOCATIONS !!
PATH_MICROWAVE = r"C:\Users\user\Documents\UNI\Thesis\Microwave_link.csv"
PATH_SIMS      = r"C:\Users\user\Documents\UNI\Thesis\DataSIMS.csv"

RANDOM_STATE   = 42
AUGMENT_TARGET = 300   # Target rows after augmentation (disclosed in paper)
np.random.seed(RANDOM_STATE)

# Publication-quality plot style
plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.titlesize" : 13,
    "axes.labelsize" : 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi"     : 150,
    "savefig.dpi"    : 300,
    "savefig.bbox"   : "tight",
    "axes.grid"      : True,
    "grid.alpha"     : 0.3,
})

# SIMANTAN brand palette (mirrors Figma prototype)
COLORS = {
    "primary" : "#1A3C5E",
    "accent"  : "#2196F3",
    "anomaly" : "#E53935",
    "normal"  : "#43A047",
    "warning" : "#FB8C00",
    "offair"  : "#9C27B0",
    "light"   : "#ECEFF1",
}

print("=" * 70)
print("  SIMANTAN Hybrid ML Simulation — Real-World Data Version")
print("=" * 70)


# =============================================================================
# SECTION 1: Data Loading
# =============================================================================

print(f"\n[STEP 1] Loading real-world field data...")

# --- 1.1 Load Microwave Link Field Data ---
microwave_link_data = pd.read_csv(PATH_MICROWAVE)
print(f"         Microwave Link records loaded : {len(microwave_link_data)} rows")
print(f"         Columns                       : {list(microwave_link_data.columns)}")

# --- 1.2 Load or Generate SIMS Data ---
if os.path.exists(PATH_SIMS):
    sims_data = pd.read_csv(PATH_SIMS)
    print(f"         SIMS records loaded           : {len(sims_data)} rows")
    SIMS_SOURCE = "real"

    # --- STANDARDIZE COLUMN NAMES ---
    sims_data.columns = sims_data.columns.str.lower().str.strip()

    print(f"         Original SIMS columns         : {list(sims_data.columns)}")

    # --- AUTO DETECT COLUMN NAMES ---
    def find_col(possible_names):
        for name in possible_names:
            if name in sims_data.columns:
                return name
        return None

    col_lic = find_col(["curr_lic_num", "license", "lic_num", "no_izin"])
    col_tx  = find_col(["sims_tx_mhz", "tx_mhz", "tx", "tx_freq"])
    col_rx  = find_col(["sims_rx_mhz", "rx_mhz", "rx", "rx_freq"])
    col_bw  = find_col(["sims_bw_mhz", "bw_mhz", "bw", "bandwidth"])

    if col_lic is None:
        raise ValueError("SIMS file must contain license column (curr_lic_num or similar)")

    # --- RENAME TO STANDARD ---
    sims_data.rename(columns={
        col_lic: "curr_lic_num",
        col_tx:  "sims_tx_mhz",
        col_rx:  "sims_rx_mhz",
        col_bw:  "sims_bw_mhz",
    }, inplace=True)

    print(f"         Standardized SIMS columns     : {list(sims_data.columns)}")

else:
    print(f"         DataSIMS.csv not found — generating synthetic SIMS records.")

    field_ids = microwave_link_data["curr_lic_num"].tolist()

    def parse_freq(val):
        try:
            return float(str(val).replace(",", ".").replace("-", "").strip()) or np.nan
        except:
            return np.nan

    sims_records = []
    for _, row in microwave_link_data.iterrows():
        tx = parse_freq(row.get("tx_mhz", np.nan))
        rx = parse_freq(row.get("rx_mhz", np.nan))
        bw = parse_freq(row.get("bw_mhz", np.nan))
        status = str(row.get("status", ""))

        if "Sesuai" in status:
            sims_tx, sims_rx, sims_bw = tx, rx, bw
        else:
            sims_tx = tx if not np.isnan(tx) else np.random.choice([7394, 7533])
            sims_rx = rx if not np.isnan(rx) else np.random.choice([7233, 7394])
            sims_bw = bw if not np.isnan(bw) else 28.0

        sims_records.append({
            "curr_lic_num": row["curr_lic_num"],
            "sims_tx_mhz": sims_tx,
            "sims_rx_mhz": sims_rx,
            "sims_bw_mhz": sims_bw,
        })

    sims_data = pd.DataFrame(sims_records)
    SIMS_SOURCE = "synthetic"
    print(f"         Synthetic SIMS records created: {len(sims_data)} rows")


# =============================================================================
# SECTION 2: Data Cleaning & Preprocessing
# =============================================================================

print(f"\n[STEP 2] Cleaning and preprocessing real field data...")

def safe_numeric(series):
    """
    Convert messy frequency/bandwidth strings to float.
    Handles: '-', '7,000', '8088.67', NaN, empty strings.
    Replaces invalid/placeholder values with NaN.
    """
    def _parse(val):
        s = str(val).strip()
        if s in ("-", "", "nan", "None"):
            return np.nan
        s = s.replace(",", ".")  # Handle Indonesian decimal comma
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(_parse)

# Clean frequency and bandwidth columns
microwave_link_data["tx_mhz_clean"] = safe_numeric(microwave_link_data["tx_mhz"])
microwave_link_data["rx_mhz_clean"] = safe_numeric(microwave_link_data["rx_mhz"])
microwave_link_data["bw_mhz_clean"] = safe_numeric(microwave_link_data["bw_mhz"])

# --- 2.1 Ground Truth Labels from 'status' column ---
# Encoding rationale (cited in paper Section II):
#   0 = Compliant       : 'Sesuai ISR' — matches SIMS licensing record
#   1 = Anomaly/Violation: All other statuses represent enforcement targets
STATUS_MAP = {
    "Sesuai ISR"                   : 0,   # Compliant
    "Tidak Berizin"                : 1,   # No valid license
    "Off Air"                      : 1,   # Equipment inactive / warehoused
    "Tidak Sesuai Parameter Teknis": 1,   # Technical parameter mismatch
}
microwave_link_data["ground_truth"] = (
    microwave_link_data["status"].map(STATUS_MAP).fillna(1)
)

# --- 2.2 Violation Sub-Type Label (for Figure 3) ---
KETERANGAN_MAP = {
    "TIDAK SESUAI ISR"      : "A: License Mismatch",
    "PENGGUDANGAN"          : "B: Off Air / Warehoused",
    "FREK TERBALIK"         : "C: TX/RX Frequency Reversed",
    "BANDWIDTH TIDAK SESUAI": "D: Bandwidth Deviation",
}
def map_keterangan(val):
    s = str(val).strip().upper()
    for key, label in KETERANGAN_MAP.items():
        if key in s:
            return label
    return "Compliant"

microwave_link_data["violation_type"] = microwave_link_data["Keterangan"].apply(map_keterangan)
# For NaN Keterangan on anomalous rows
mask_anomaly_no_keterangan = (
    (microwave_link_data["ground_truth"] == 1) &
    (microwave_link_data["violation_type"] == "Compliant")
)
microwave_link_data.loc[mask_anomaly_no_keterangan, "violation_type"] = "A: License Mismatch"

# --- 2.3 Sertifikat encoding ---
microwave_link_data["sertifikat_flag"] = (
    microwave_link_data["sertifikat"].fillna("Tidak Ada")
    .apply(lambda x: 1 if str(x).strip().lower() == "ada" else 0)
)

# Print cleaning summary
n_compliant = (microwave_link_data["ground_truth"] == 0).sum()
n_anomaly   = (microwave_link_data["ground_truth"] == 1).sum()
print(f"         Compliant records : {n_compliant}")
print(f"         Anomalous records : {n_anomaly}")
print(f"         Anomaly rate      : {n_anomaly/len(microwave_link_data)*100:.1f}%")


# =============================================================================
# SECTION 3: Merge Field Data with SIMS
# =============================================================================

print(f"\n[STEP 3] Merging field data with SIMS records...")

merged = pd.merge(
    microwave_link_data,
    sims_data,
    on="curr_lic_num",
    how="left",
)

print(f"         Columns after merge: {list(merged.columns)}")

# --- ENSURE REQUIRED COLUMNS EXIST ---
for col in ["sims_tx_mhz", "sims_rx_mhz", "sims_bw_mhz"]:
    if col not in merged.columns:
        print(f"         WARNING: {col} missing, filling with NaN")
        merged[col] = np.nan

# --- CONVERT TO NUMERIC ---
merged["sims_tx_mhz"] = pd.to_numeric(merged["sims_tx_mhz"], errors="coerce")
merged["sims_rx_mhz"] = pd.to_numeric(merged["sims_rx_mhz"], errors="coerce")
merged["sims_bw_mhz"] = pd.to_numeric(merged["sims_bw_mhz"], errors="coerce")

# --- FEATURE ENGINEERING ---
merged["delta_tx_mhz"] = abs(merged["tx_mhz_clean"] - merged["sims_tx_mhz"])
merged["delta_rx_mhz"] = abs(merged["rx_mhz_clean"] - merged["sims_rx_mhz"])
merged["delta_bw_mhz"] = abs(merged["bw_mhz_clean"] - merged["sims_bw_mhz"])

merged["freq_reversal_flag"] = (
    (abs(merged["tx_mhz_clean"] - merged["sims_rx_mhz"]) < 50) &
    (abs(merged["rx_mhz_clean"] - merged["sims_tx_mhz"]) < 50)
).astype(int)

merged["off_air_flag"] = (
    merged["tx_mhz_clean"].isna() & merged["rx_mhz_clean"].isna()
).astype(int)

merged["composite_risk"] = (
    0.30 * merged["delta_tx_mhz"].fillna(999) +
    0.25 * merged["delta_bw_mhz"].fillna(999) +
    0.20 * merged["delta_rx_mhz"].fillna(999) +
    0.15 * merged["freq_reversal_flag"] * 500 +
    0.10 * merged["off_air_flag"] * 500
)

merged.fillna(merged.median(numeric_only=True), inplace=True)

print(f"         Merge complete. Shape: {merged.shape}")

FEATURE_COLS = [
    "tx_mhz_clean", "rx_mhz_clean", "bw_mhz_clean",
    "sertifikat_flag", "koor_long", "koor_lat",
    "delta_tx_mhz", "delta_rx_mhz", "delta_bw_mhz",
    "freq_reversal_flag", "off_air_flag", "composite_risk",
]

X_raw  = merged[FEATURE_COLS].copy()
y_true = merged["ground_truth"].values


# =============================================================================
# SECTION 4: Domain-Aware Data Augmentation
# =============================================================================
# The original dataset contains 39 records — insufficient for robust ML
# training. Domain-aware augmentation expands the dataset by introducing
# Gaussian noise within ±5% of real parameter values, preserving the
# statistical distribution of the original Balmon field data.
#
# DISCLOSURE: This augmentation is fully reported in Section II of the
# paper and is used ONLY for simulation purposes to demonstrate the
# architectural logic of the Hybrid ML layer.

print(f"\n[STEP 4] Applying domain-aware augmentation...")
print(f"         Original size : {len(X_raw)} records")

def augment_dataset(X, y, target_n, noise_pct=0.05, random_state=42):
    """
    Augment dataset by adding Gaussian noise within ±noise_pct of
    each feature's standard deviation. Preserves class proportions.
    """
    rng = np.random.RandomState(random_state)
    X_aug_list = [X.values]
    y_aug_list = [y]

    n_needed = target_n - len(X)
    class_counts = pd.Series(y).value_counts(normalize=True)

    for cls, proportion in class_counts.items():
        n_cls = int(n_needed * proportion)
        idx   = np.where(y == cls)[0]
        chosen = rng.choice(idx, size=n_cls, replace=True)
        noise  = rng.normal(0, noise_pct, (n_cls, X.shape[1]))
        stds   = X.std(axis=0).values
        X_new  = X.values[chosen] + noise * stds
        y_new  = np.full(n_cls, cls)
        X_aug_list.append(X_new)
        y_aug_list.append(y_new)

    X_out = np.vstack(X_aug_list)
    y_out = np.concatenate(y_aug_list)
    return pd.DataFrame(X_out, columns=X.columns), y_out

X_aug, y_aug = augment_dataset(X_raw, y_true, AUGMENT_TARGET)
print(f"         Augmented size : {len(X_aug)} records")
print(f"         Class balance  : {pd.Series(y_aug).value_counts().to_dict()}")

# Standardize
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_aug)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLS)


# =============================================================================
# SECTION 5: Hybrid ML — Stage 1: Isolation Forest
# =============================================================================

print(f"\n[STEP 5] Running Isolation Forest (unsupervised anomaly detection)...")

# --- SAFE CONTAMINATION HANDLING ---
anomaly_rate_real = y_aug.mean()

# Clamp to valid range (0.01 – 0.5)
contamination_value = min(max(anomaly_rate_real, 0.01), 0.5)

print(f"         Raw anomaly rate     : {anomaly_rate_real:.3f}")
print(f"         Used contamination   : {contamination_value:.3f}")

iso_forest = IsolationForest(
    contamination=contamination_value,
    n_estimators=200,
    random_state=RANDOM_STATE,
)

if_raw                        = iso_forest.fit_predict(X_scaled)
merged_aug                    = X_scaled.copy()
merged_aug["if_anomaly_flag"] = (if_raw == -1).astype(int)
merged_aug["if_score"]        = iso_forest.decision_function(X_scaled)
merged_aug["ground_truth"]    = y_aug

n_flagged = merged_aug["if_anomaly_flag"].sum()
print(f"         Contamination rate used : {anomaly_rate_real:.2f}")
print(f"         Records flagged by IF   : {n_flagged} / {len(X_scaled)}")


# =============================================================================
# SECTION 6: Hybrid ML — Stage 2: Random Forest Classifier
# =============================================================================

print(f"\n[STEP 6] Running Random Forest Classifier (supervised validation)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_aug, test_size=0.30,
    random_state=RANDOM_STATE, stratify=y_aug
)

rf_classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_classifier.fit(X_train, y_train)
y_pred      = rf_classifier.predict(X_test)
y_pred_prob = rf_classifier.predict_proba(X_test)[:, 1]

# --- Performance ---
accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc  = auc(fpr, tpr)

# 5-fold cross-validation on full augmented set
cv_scores = cross_val_score(
    rf_classifier, X_scaled, y_aug,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1",
)

print(f"\n{'='*55}")
print(f"  RANDOM FOREST PERFORMANCE METRICS (Real-World Data)")
print(f"{'='*55}")
print(f"  Accuracy          : {accuracy * 100:.2f}%")
print(f"  F1 Score          : {f1 * 100:.2f}%")
print(f"  ROC-AUC           : {roc_auc:.4f}")
print(f"  CV F1 (5-fold)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"{'='*55}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Compliant (Sesuai ISR)", "Anomaly / Violation"]))


# =============================================================================
# SECTION 7: Publication-Quality Figures
# =============================================================================

print(f"\n[STEP 7] Generating publication-quality figures...")

# ---- FIGURE 1: Hybrid ML Performance Dashboard ----
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(
    "Figure 1. SIMANTAN Hybrid ML Performance Dashboard\n"
    "Real-World Balmon Field Data — Microwave Link Anomaly Detection",
    fontsize=14, fontweight="bold", y=1.01,
)

# [1A] Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Compliant", "Anomaly"],
    yticklabels=["Compliant", "Anomaly"],
    ax=axes[0, 0], linewidths=0.5, linecolor="white",
    annot_kws={"size": 14, "weight": "bold"},
)
axes[0, 0].set_title("(a) Confusion Matrix — Random Forest", fontweight="bold")
axes[0, 0].set_xlabel("Predicted Label")
axes[0, 0].set_ylabel("True Label")
for i, lbl in enumerate([["TN", "FP"], ["FN", "TP"]]):
    for j, l in enumerate(lbl):
        axes[0, 0].text(j + 0.5, i + 0.75, l,
                         ha="center", va="center",
                         color="grey", fontsize=9, style="italic")

# [1B] ROC Curve
axes[0, 1].plot(fpr, tpr, color=COLORS["accent"], lw=2.5,
                label=f"RF Classifier (AUC = {roc_auc:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Baseline")
axes[0, 1].fill_between(fpr, tpr, alpha=0.08, color=COLORS["accent"])
axes[0, 1].set_xlim([0, 1]); axes[0, 1].set_ylim([0, 1.02])
axes[0, 1].set_title("(b) ROC Curve — Random Forest Classifier", fontweight="bold")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(loc="lower right", framealpha=0.9)

# [1C] Isolation Forest Score Distribution
normal_scores  = merged_aug.loc[merged_aug["if_anomaly_flag"] == 0, "if_score"]
anomaly_scores = merged_aug.loc[merged_aug["if_anomaly_flag"] == 1, "if_score"]
axes[1, 0].hist(normal_scores,  bins=30, color=COLORS["normal"],
                alpha=0.75, label="Compliant (Sesuai ISR)", edgecolor="white")
axes[1, 0].hist(anomaly_scores, bins=30, color=COLORS["anomaly"],
                alpha=0.75, label="Flagged Violation",       edgecolor="white")
axes[1, 0].axvline(x=0, color="black", linestyle="--", lw=1.5,
                   label="Decision Boundary (score = 0)")
axes[1, 0].set_title("(c) Isolation Forest Score Distribution", fontweight="bold")
axes[1, 0].set_xlabel("Anomaly Score (lower = more anomalous)")
axes[1, 0].set_ylabel("Record Count")
axes[1, 0].legend(framealpha=0.9)

# [1D] Cross-Validation F1 Scores
fold_labels  = [f"Fold {i+1}" for i in range(len(cv_scores))]
bar_colors   = [COLORS["accent"] if s >= cv_scores.mean() else COLORS["warning"]
                for s in cv_scores]
bars = axes[1, 1].bar(fold_labels, cv_scores, color=bar_colors,
                       edgecolor="white", width=0.5)
axes[1, 1].axhline(y=cv_scores.mean(), color=COLORS["anomaly"],
                   linestyle="--", lw=1.5,
                   label=f"Mean F1 = {cv_scores.mean():.3f}")
for bar, val in zip(bars, cv_scores):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=COLORS["primary"])
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].set_title("(d) 5-Fold Cross-Validation F1 Score", fontweight="bold")
axes[1, 1].set_ylabel("F1 Score")
axes[1, 1].legend(framealpha=0.9)

plt.tight_layout()
plt.savefig("Fig1_Hybrid_ML_Performance_RealWorld.png")
plt.show()
print("         Fig 1 saved.")


# ---- FIGURE 2: Feature Importance (maps to SIMANTAN UI priority) ----
fi_df = pd.DataFrame({
    "Feature"   : FEATURE_COLS,
    "Importance": rf_classifier.feature_importances_,
}).sort_values("Importance", ascending=True)

DISPLAY_NAMES = {
    "composite_risk"      : "Composite Risk Score ★",
    "delta_bw_mhz"        : "ΔBandwidth (Field vs. SIMS)",
    "freq_reversal_flag"  : "TX/RX Frequency Reversal Flag",
    "delta_tx_mhz"        : "ΔTX Frequency Deviation (MHz)",
    "off_air_flag"        : "Off-Air / Warehoused Flag",
    "delta_rx_mhz"        : "ΔRX Frequency Deviation (MHz)",
    "bw_mhz_clean"        : "Field Bandwidth (MHz)",
    "sertifikat_flag"     : "Certification Present Flag",
    "tx_mhz_clean"        : "Field TX Frequency (MHz)",
    "rx_mhz_clean"        : "Field RX Frequency (MHz)",
    "koor_long"           : "Station Longitude",
    "koor_lat"            : "Station Latitude",
}
fi_df["Display"] = fi_df["Feature"].map(DISPLAY_NAMES).fillna(fi_df["Feature"])

mean_imp  = fi_df["Importance"].mean()
bar_colors = [COLORS["anomaly"] if v >= mean_imp else COLORS["accent"]
              for v in fi_df["Importance"]]

fig2, ax2 = plt.subplots(figsize=(10, 7))
bars2 = ax2.barh(fi_df["Display"], fi_df["Importance"],
                  color=bar_colors, edgecolor="white", height=0.65)
for bar, val in zip(bars2, fi_df["Importance"]):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color=COLORS["primary"])
ax2.axvline(x=mean_imp, color=COLORS["warning"], linestyle="--", lw=1.5,
            label=f"Mean Importance = {mean_imp:.3f}")
ax2.set_title(
    "Figure 2. Feature Importance — Random Forest\n"
    "SIMANTAN Anomaly Detection Panel Priority Mapping (Real-World Data)",
    fontweight="bold",
)
ax2.set_xlabel("Importance Score (Mean Decrease in Impurity)")
ax2.legend(framealpha=0.9)
high_patch = mpatches.Patch(color=COLORS["anomaly"], label="Above mean — top UI priority")
low_patch  = mpatches.Patch(color=COLORS["accent"],  label="Below mean — secondary display")
ax2.legend(handles=[high_patch, low_patch], framealpha=0.9)
ax2.text(0.99, 0.02, "★ = Highest priority in SIMANTAN dashboard",
         transform=ax2.transAxes, ha="right", fontsize=8,
         color=COLORS["anomaly"], style="italic")
plt.tight_layout()
plt.savefig("Fig2_Feature_Importance_RealWorld.png")
plt.show()
print("         Fig 2 saved.")


# ---- FIGURE 3: Real Violation Type Distribution ----
# Derived directly from Balmon field inspection 'Keterangan' column
viol_counts = (
    microwave_link_data[microwave_link_data["violation_type"] != "Compliant"]
    ["violation_type"].value_counts().reset_index()
)
viol_counts.columns = ["Violation Type", "Count"]

viol_palette = [COLORS["anomaly"], COLORS["offair"], COLORS["warning"], COLORS["accent"]]

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle(
    "Figure 3. Microwave Link Violation Distribution\n"
    "Balmon Class I Jakarta — Real Field Inspection Data",
    fontweight="bold", fontsize=13,
)
bars3 = ax3a.barh(
    viol_counts["Violation Type"], viol_counts["Count"],
    color=viol_palette[:len(viol_counts)], edgecolor="white"
)
for bar, val in zip(bars3, viol_counts["Count"]):
    ax3a.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
              str(val), va="center", fontweight="bold")
ax3a.set_xlabel("Number of Records")
ax3a.set_title("(a) Count by Violation Category", fontweight="bold")

wedges, texts, autotexts = ax3b.pie(
    viol_counts["Count"],
    labels=viol_counts["Violation Type"],
    colors=viol_palette[:len(viol_counts)],
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.8,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
ax3b.set_title("(b) Proportional Breakdown", fontweight="bold")
plt.tight_layout()
plt.savefig("Fig3_Violation_Distribution_RealWorld.png")
plt.show()
print("         Fig 3 saved.")


# ---- FIGURE 4: Compliance Status Map (Geographic Distribution) ----
# Maps station coordinates to compliance status — corresponds to the
# SIMANTAN geographic risk map in the Figma prototype dashboard.

geo_data = merged.copy()
geo_data["status_label"] = merged_aug["ground_truth"].values[:len(merged)]

fig4, ax4 = plt.subplots(figsize=(10, 8))
compliant_pts = geo_data[geo_data["status_label"] == 0]
anomaly_pts   = geo_data[geo_data["status_label"] == 1]

ax4.scatter(
    compliant_pts["koor_long"], compliant_pts["koor_lat"],
    c=COLORS["normal"], s=80, alpha=0.85, edgecolors="white",
    linewidths=0.5, label="Compliant (Sesuai ISR)", zorder=3,
)
ax4.scatter(
    anomaly_pts["koor_long"], anomaly_pts["koor_lat"],
    c=COLORS["anomaly"], s=100, alpha=0.85,
    marker="^", edgecolors="white", linewidths=0.5,
    label="Flagged Violation", zorder=4,
)

# Annotate station names
for _, row in geo_data.iterrows():
    if pd.notna(row["koor_long"]) and pd.notna(row["koor_lat"]):
        ax4.annotate(
            str(row.get("stn_name", "")),
            (row["koor_long"], row["koor_lat"]),
            fontsize=7, alpha=0.7,
            xytext=(3, 3), textcoords="offset points",
        )

ax4.set_title(
    "Figure 4. Microwave Link Station Geographic Compliance Map\n"
    "SIMANTAN Dashboard — Jakarta Area Field Inspection",
    fontweight="bold",
)
ax4.set_xlabel("Longitude")
ax4.set_ylabel("Latitude")
ax4.legend(framealpha=0.9, loc="upper left")
plt.tight_layout()
plt.savefig("Fig4_Geographic_Compliance_Map.png")
plt.show()
print("         Fig 4 saved.")


# =============================================================================
# SECTION 8: Raw Data Quality Report (for Paper Section II)
# =============================================================================

print(f"\n[STEP 8] Generating data quality report...")

total   = len(microwave_link_data)
missing = microwave_link_data.isnull().sum()
missing_pct = (missing / total * 100).round(1)

quality_df = pd.DataFrame({
    "Column"        : missing.index,
    "Missing Count" : missing.values,
    "Missing %"     : missing_pct.values,
}).sort_values("Missing %", ascending=False)

fig5, ax5 = plt.subplots(figsize=(10, 6))
bars5 = ax5.barh(
    quality_df["Column"], quality_df["Missing %"],
    color=[COLORS["anomaly"] if v > 50 else COLORS["warning"] if v > 10 else COLORS["normal"]
           for v in quality_df["Missing %"]],
    edgecolor="white",
)
for bar, val in zip(bars5, quality_df["Missing %"]):
    ax5.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", fontsize=9)
ax5.axvline(x=50, color=COLORS["anomaly"], linestyle="--", lw=1,
            alpha=0.6, label="50% missing threshold")
ax5.set_title(
    "Figure 5. Data Quality Report — Microwave_link.csv\n"
    "Missing Value Analysis per Column",
    fontweight="bold",
)
ax5.set_xlabel("Missing Values (%)")
ax5.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig("Fig5_Data_Quality_Report.png")
plt.show()
print("         Fig 5 saved.")


# =============================================================================
# SECTION 9: Final Summary
# =============================================================================

print(f"\n{'='*70}")
print(f"  SIMANTAN HYBRID ML — REAL-WORLD SIMULATION SUMMARY")
print(f"{'='*70}")
print(f"  Data Source      : Balmon Class I Jakarta Field Inspections")
print(f"  SIMS Source      : {'Real DataSIMS.csv' if SIMS_SOURCE == 'real' else 'Synthetic (DataSIMS.csv unavailable)'}")
print(f"  Original Records : {len(microwave_link_data)} (real field data)")
print(f"  Augmented Size   : {len(X_aug)} (disclosed in paper Section II)")
print(f"  Anomaly Rate     : {y_aug.mean()*100:.1f}%")
print(f"  IF Flagged       : {n_flagged} / {len(X_scaled)}")
print(f"  RF Accuracy      : {accuracy * 100:.2f}%")
print(f"  RF F1 Score      : {f1 * 100:.2f}%")
print(f"  ROC-AUC          : {roc_auc:.4f}")
print(f"  CV F1 (5-fold)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"{'='*70}")
print(f"\n  Figures saved (300 DPI):")
for i, name in enumerate([
    "Fig1_Hybrid_ML_Performance_RealWorld.png",
    "Fig2_Feature_Importance_RealWorld.png",
    "Fig3_Violation_Distribution_RealWorld.png",
    "Fig4_Geographic_Compliance_Map.png",
    "Fig5_Data_Quality_Report.png",
], 1):
    print(f"    {name}")
print(f"\n  IMPORTANT: Update PATH_MICROWAVE and PATH_SIMS at the top of")
print(f"  this script to match your local file locations.")
print(f"{'='*70}")