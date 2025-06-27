import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump

# ------------------------------
# STEP 1: Generate Simulated Data
# ------------------------------

def simulate_fall_data(n_normal=2000, n_fall=500):
    data = []

    # Normal movements (e.g., walking, sitting)
    for _ in range(n_normal):
        ax = np.random.normal(0, 1.5)
        ay = np.random.normal(0, 1.5)
        az = np.random.normal(9.8, 1.5)  # gravity effect
        label = 0
        data.append([ax, ay, az, label])

    # Falls (sudden abnormal spikes)
    for _ in range(n_fall):
        ax = np.random.normal(10, 5)
        ay = np.random.normal(10, 5)
        az = np.random.normal(0, 5)
        label = 1
        data.append([ax, ay, az, label])

    df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'fall'])
    return df.sample(frac=1).reset_index(drop=True)

df = simulate_fall_data()

# ------------------------------
# STEP 2: Feature Engineering
# ------------------------------

# Add magnitude of acceleration as feature
df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

# ------------------------------
# STEP 3: Data Preprocessing
# ------------------------------

X = df[['ax', 'ay', 'az', 'acc_mag']]
y = df['fall']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
dump(scaler, 'fall_scaler.joblib')

# ------------------------------
# STEP 4: Train/Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# ------------------------------
# STEP 5: Model Training
# ------------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
dump(model, 'fall_model.joblib')

# ------------------------------
# STEP 6: Evaluation
# ------------------------------

y_pred = model.predict(X_test)

print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred))

print("✅ Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fall'], yticklabels=['Normal', 'Fall'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ------------------------------
# STEP 7: Feature Importance
# ------------------------------

importances = model.feature_importances_
features = ['ax', 'ay', 'az', 'acc_mag']

plt.figure(figsize=(6,4))
sns.barplot(x=features, y=importances)
plt.title("Feature Importance")
plt.show()

# ------------------------------
# STEP 8: Predict New Sample
# ------------------------------

def predict_fall(ax, ay, az):
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    sample = scaler.transform([[ax, ay, az, acc_mag]])
    pred = model.predict(sample)
    return "FALL DETECTED" if pred[0] == 1 else "Normal Movement"

# Example usage
print("\n📌 Test New Sample:")
print(predict_fall(12, 13, 2))
print(predict_fall(0.5, -0.3, 9.2))
