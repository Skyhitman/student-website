# train_and_save_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1) Generate synthetic dataset (your formula)
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'study_hours': np.random.uniform(0, 6, n_samples),
    'attendance_percent': np.random.uniform(40, 100, n_samples),
    'past_score': np.random.uniform(30, 100, n_samples),
    'social_media_hours': np.random.uniform(0, 6, n_samples)
})

data['pass_fail'] = (
    (data['study_hours'] * 10 +
     data['attendance_percent'] * 0.3 +
     data['past_score'] * 0.4 -
     data['social_media_hours'] * 5) > 100
).astype(int)

# 2) Train / test split
X = data[['study_hours', 'attendance_percent', 'past_score', 'social_media_hours']]
y = data['pass_fail']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3) Train Random Forest with tuned hyperparams
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 4) Evaluate
y_pred = rf.predict(X_test)
print("Validation Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5) Save model
joblib.dump(rf, "student_success_model.pkl")
print("Saved model to student_success_model.pkl")
