import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
microwave_link_data = pd.read_csv('C:\\Users\\user\\Documents\\UNI\\Thesis\\Microwave_link.csv')
sims_data = pd.read_csv('C:\\Users\\user\\Documents\\UNI\\Thesis\\DataSIMS.csv')

# Display the first few rows of each DataFrame to verify
print("Microwave Link Data:")
print(microwave_link_data.head())
print("\nSIMS Data:")
print(sims_data.head())

# Merge the data on 'curr_lic_num'
merged_data = pd.merge(microwave_link_data, sims_data, left_on='curr_lic_num', right_on='curr_lic_num', how='left')

# Display the merged DataFrame to verify the results
print("\nMerged Data:")
print(merged_data.head())

# Preprocessing steps
# Convert categorical variables to numerical using LabelEncoder
label_encoders = {}
for column in merged_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    merged_data[column] = le.fit_transform(merged_data[column])
    label_encoders[column] = le

# Fill missing values with the median of each column
merged_data.fillna(merged_data.median(), inplace=True)

# Standardize numerical features
scaler = StandardScaler()
numerical_features = merged_data.select_dtypes(include=['float64', 'int64']).columns
merged_data[numerical_features] = scaler.fit_transform(merged_data[numerical_features])

# Split the data into features and target (anomaly detection, so no explicit target)
X = merged_data.drop(columns=['curr_lic_num'])  # Assuming 'curr_lic_num' is not needed for ML

# Anomaly Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
merged_data['anomaly'] = iso_forest.fit_predict(X)

# Display the results of anomaly detection
print("\nAnomaly Detection Results:")
print(merged_data[['curr_lic_num', 'anomaly']].head())

# Visualize the anomalies using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(merged_data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Random Forest Classification (for demonstration purposes)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
y = merged_data['anomaly']  # Use 'anomaly' as the target for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# Evaluate the Random Forest Classifier
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
