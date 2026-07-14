import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

cutoff = df["date"].quantile(0.8)
train = df[df["date"] < cutoff]
test = df[df["date"] >= cutoff]

print(f"Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"Test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")

drop_cols = ["team", "team_opp", "date", "season", "won"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X_train, y_train = train[feature_cols].copy(), train["won"]
X_test, y_test = test[feature_cols].copy(), test["won"]

train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

baseline_pred = test["home"] == 1
baseline_acc = accuracy_score(y_test, baseline_pred)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
model_acc = accuracy_score(y_test, model.predict(X_test_scaled))

print("\n--- Results ---")
print(f"Baseline ('home team always wins') accuracy: {baseline_acc:.3f}")
print(f"Logistic regression accuracy:                {model_acc:.3f}")

if model_acc > baseline_acc:
    print(f"\n✅ Your features beat the naive baseline by {model_acc - baseline_acc:.3f}. Good signal — safe to hand off.")
else:
    print("\n⚠️ Your features do NOT beat the naive baseline. Something may need revisiting before handoff.")

coefs = pd.Series(model.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
print("\nTop 8 most influential features:")
print(coefs.head(8))