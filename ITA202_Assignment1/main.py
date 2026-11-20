
# ITA202 - ASSIGNMENT 1
# Du doan kha nang song sot tren tau Titanic

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


# 1. Doc du lieu
try:
    df = pd.read_csv("titanic.csv")
except FileNotFoundError:
    raise SystemExit("Khong tim thay file titanic.csv!")

print("=== DU LIEU GOC ===")
print("Shape:", df.shape)
df.info()
print(df.describe())

print("\nGia tri thieu:")
print(df.isna().sum())


# 2. Chon cac cot du lieu can thiet
cols = [
    "Survived", "Pclass", "Sex", "Age",
    "SibSp", "Parch", "Fare", "Embarked"
]

df = df[cols].copy()


# 3. Chia du lieu Train 80% va Test 20%
X = df.drop(columns="Survived")
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X_train.copy()
X_test = X_test.copy()


# 4. Xu ly du lieu thieu
for col in ["Age", "Fare"]:
    value = X_train[col].median()
    X_train[col] = X_train[col].fillna(value)
    X_test[col] = X_test[col].fillna(value)

for col in ["Sex", "Embarked"]:
    value = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(value)
    X_test[col] = X_test[col].fillna(value)


# 5. Xu ly gia tri ngoai le bang IQR
q1 = X_train["Fare"].quantile(0.25)
q3 = X_train["Fare"].quantile(0.75)

iqr = q3 - q1
lower = max(0, q1 - 1.5 * iqr)
upper = q3 + 1.5 * iqr

print("\nSo gia tri ngoai le:",
      ((X_train["Fare"] < lower) |
       (X_train["Fare"] > upper)).sum())

X_train["Fare"] = X_train["Fare"].clip(lower, upper)
X_test["Fare"] = X_test["Fare"].clip(lower, upper)


# 6. Chuyen du lieu chu thanh so
X_train = pd.get_dummies(
    X_train, columns=["Sex", "Embarked"], dtype=int
)

X_test = pd.get_dummies(
    X_test, columns=["Sex", "Embarked"], dtype=int
)

X_test = X_test.reindex(
    columns=X_train.columns, fill_value=0
)


# 7. Chuan hoa du lieu
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 8. Luu du lieu da tien xu ly
train_result = pd.DataFrame(
    X_train_scaled, columns=X_train.columns
)

train_result["Survived"] = y_train.to_numpy()
train_result["Split"] = "train"

test_result = pd.DataFrame(
    X_test_scaled, columns=X_train.columns
)

test_result["Survived"] = y_test.to_numpy()
test_result["Split"] = "test"

processed = pd.concat(
    [train_result, test_result],
    ignore_index=True
)

processed.to_csv("titanic_processed.csv", index=False)

print("\n=== DU LIEU DA XU LY ===")
print("Train:", X_train_scaled.shape)
print("Test:", X_test_scaled.shape)
print("So o thieu:", processed.isna().sum().sum())

print(processed.head())
print("Da luu titanic_processed.csv")


# 9. Huan luyen mo hinh Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


# 10. Danh gia mo hinh
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))


# 11. Ve bieu do Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Titanic - Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=200)

print("Da luu confusion_matrix.png")

plt.show()
