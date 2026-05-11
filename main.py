import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("predictive_maintenance_dataset.csv")

# Display first rows
print(df.head())

# Remove non-useful columns
df = df.drop(["date", "device"], axis=1)

# Features and target
X = df.drop("failure", axis=1)
y = df["failure"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Failure distribution graph
df["failure"].value_counts().plot(kind="bar")

plt.title("Failure Distribution")
plt.xlabel("Failure")
plt.ylabel("Count")

plt.show()

# Feature importance

importance = model.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_df)

# Plot feature importance

plt.figure(figsize=(10,5))

plt.bar(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.show()