import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Load the cleaned dataset
df = pd.read_csv(os.path.join('..', 'data', 'processed', 'cleaned.csv'))

# Separate target column
y = df['fraud_reported']

# Select numeric features and drop columns with all NaNs
X = df.drop(columns=['fraud_reported']).select_dtypes(include='number')
X = X.dropna(axis=1, how='all')  # Drop completely empty columns

# Impute missing values using median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

#  Confirm data is clean
print("Any NaNs left in X after imputation?", X.isna().sum().sum())
print(" Final shape of X:", X.shape)

#  show class distribution before SMOTE
print("\n Original class distribution:\n", y.value_counts())

#  Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\n Resampled class distribution:\n", y_resampled.value_counts())

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

#  Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Evaluate model
y_pred = model.predict(X_test)
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

#  Save model to file
model_path = os.path.join('..', 'models', 'fraud_model.pkl')
joblib.dump(model, model_path)
print(f"\n SMOTE-enhanced model saved to: {model_path}")
