import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
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

# Check if 'tanggal' column exists in microwave_link_data
if 'tanggal' not in microwave_link_data.columns:
    print("Error: 'tanggal' column is missing from Microwave Link data. Can't be calculated.")
else:
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
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Display the percentage of accuracy and F1 score
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Visualize the confusion matrix with larger figure size
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Random Forest')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

        # Feature Importance
    feature_importances = rf_classifier.feature_importances_
    features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    features_df = features_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(features_df)

        # Visualize Feature Importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.title('Feature Importance')
    plt.show()

# Display mismatched data
mismatched_data = microwave_link_data[~microwave_link_data['curr_lic_num'].isin(sims_data['curr_lic_num'])]

print("\nMismatched Data:")
print(mismatched_data)

# Filter out the "Sesuai ISR" value from the 'status' column
filtered_mismatched_data = mismatched_data[mismatched_data['status'] != 'Sesuai ISR']

# Count the occurrences of each status
status_counts = filtered_mismatched_data['status'].value_counts()

# Visualize using a bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x=status_counts.values, y=status_counts.index, palette='viridis')
plt.title('Distribution of Status in Mismatched Data (Excluding "Sesuai ISR")')
plt.xlabel('Count')
plt.ylabel('Status')
plt.show()


# # Count the occurrences of each 'status' in the mismatched data
# status_counts = mismatched_data['status'].value_counts()

# # Visualize the status counts with a bar plot
# plt.figure(figsize=(10, 6))
# sns.barplot(x=status_counts.index, y=status_counts.values, palette='viridis')
# plt.title('Mismatched Data by Status')
# plt.xlabel('Status')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.show()



# # Assuming 'category_column' is a categorical column in your mismatched data
# for column in mismatched_data.select_dtypes(include=['object']).columns:
#     plt.figure(figsize=(12, 8))
#     sns.countplot(y=column, data=mismatched_data)
#     plt.title(f'Distribution of {column} in Mismatched Data')
#     plt.show()

# # Visualize the mismatched data
# plt.figure(figsize=(12, 8))
# sns.pairplot(mismatched_data)
# plt.suptitle('Mismatched Data Pairplot', y=1.02)
# plt.show()

