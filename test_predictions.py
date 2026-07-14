"""
Deeper check on whether the model is really predicting anything.
Run this after build_features.py. Complements quick_signal_test.py.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

cutoff = df["date"].quantile(0.8)
train = df[df["date"] < cutoff]
test = df[df["date"] >= cutoff].copy()

drop_cols = ["team", "team_opp", "date", "season", "won"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X_train, y_train = train[feature_cols].copy(), train["won"]
X_test, y_test = test[feature_cols].copy(), test["won"]

train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

print("=== 1. Confusion matrix ===")
cm = confusion_matrix(y_test, preds)
print(f"                Predicted Loss  Predicted Win")
print(f"Actual Loss     {cm[0][0]:>13}  {cm[0][1]:>13}")
print(f"Actual Win      {cm[1][0]:>13}  {cm[1][1]:>13}")
print("(A model that just guesses the majority class would show one column near-empty. "
      "Balanced numbers here mean it's genuinely discriminating, not defaulting.)")

print("\n=== 2. ROC-AUC (ranking quality, threshold-independent) ===")
auc = roc_auc_score(y_test, probs)
print(f"AUC: {auc:.3f}  (0.5 = random guessing, 1.0 = perfect. >0.70 is solid for NBA games.)")

print("\n=== 3. Calibration check: does '70% confident' actually mean ~70% win rate? ===")
test["pred_prob"] = probs
test["bucket"] = pd.cut(test["pred_prob"], bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
calib = test.groupby("bucket", observed=True).agg(
    n=("won", "size"),
    predicted_avg=("pred_prob", "mean"),
    actual_win_rate=("won", "mean"),
)
print(calib.round(3))
print("(If actual_win_rate tracks predicted_avg reasonably closely per bucket, "
      "the model's confidence is meaningful, not just noise.)")

print("\n=== 4. Spot-check: 5 individual predicted games ===")
sample = test.sample(5, random_state=42)
sample_idx = sample.index
sample_X = scaler.transform(sample[feature_cols].fillna(train_means))
sample_probs = model.predict_proba(sample_X)[:, 1]
for i, (idx, row) in enumerate(sample.iterrows()):
    actual = "WIN" if row["won"] else "LOSS"
    predicted = "WIN" if sample_probs[i] > 0.5 else "LOSS"
    match = "✅" if predicted == actual else "❌"
    print(f"{row['date'].date()}  {row['team']} vs {row['team_opp']}  "
          f"| model says {predicted} ({sample_probs[i]:.0%} conf)  | actually {actual}  {match}")