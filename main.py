import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from itertools import cycle

# LOAD DATASET DAN EDA (Eksplorasi Data)
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Konversi ke DataFrame Pandas untuk EDA yang lebih mudah
df = pd.DataFrame(X, columns=feature_names)
df["species_id"] = y
df["species_name"] = df["species_id"].map(lambda x: target_names[x])

print("Informasi Dataset:")
df.info()

print("\nStatistik Deskriptif:")
print(df.describe())

print("\nDistribusi Kelas:")
print(df["species_name"].value_counts())

# Visualisasi EDA Sederhana (Pairplot)
print("\nMenampilkan Pairplot untuk EDA...")
sns.pairplot(df, hue="species_name", markers=["o", "s", "D"])
plt.suptitle("Pairplot Dataset Iris", y=1.02)
plt.show()


# PREPROCESSING
# Pisahkan data menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Lakukan Scaling (Standardisasi)
# Penting untuk Logistic Regression & SVM, tidak wajib untuk Decision Tree
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nUkuran data latih: {X_train_scaled.shape}")
print(f"Ukuran data uji: {X_test_scaled.shape}")


# 4. MODELING (Minimal 2 Algoritma)
# --- Model 1: Logistic Regression ---
print("\n--- Melatih Model 1: Logistic Regression ---")
# 'multi_class='ovr'' (One-vs-Rest) cocok untuk masalah ini
model_lr = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
y_prob_lr = model_lr.predict_proba(X_test_scaled)


# --- Model 2: Decision Tree ---
print("--- Melatih Model 2: Decision Tree ---")
model_dt = DecisionTreeClassifier(random_state=42, max_depth=4)
# Decision Tree tidak sensitif terhadap skala fitur,
# tapi kita gunakan data scaled agar konsisten
model_dt.fit(X_train_scaled, y_train)
y_pred_dt = model_dt.predict(X_test_scaled)
y_prob_dt = model_dt.predict_proba(X_test_scaled)


# EVALUASI MODEL
print("\n=============================================")
print("  HASIL EVALUASI: LOGISTIC REGRESSION")
print("=============================================")

# a. Confusion Matrix - Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix (Logistic Regression):\n", cm_lr)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# b. Accuracy, Precision, Recall, F1-score - Logistic Regression
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr, target_names=target_names))


print("\n=============================================")
print("    HASIL EVALUASI: DECISION TREE")
print("=============================================")

# a. Confusion Matrix - Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix (Decision Tree):\n", cm_dt)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# b. Accuracy, Precision, Recall, F1-score - Decision Tree
print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt, target_names=target_names))


# -------------------------------------------------------------------
# c. ROC Curve (untuk Multiclass)
# -------------------------------------------------------------------
# Kita perlu binarize y_test
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# --- ROC untuk Logistic Regression ---
fpr_lr = dict()
tpr_lr = dict()
roc_auc_lr = dict()
for i in range(n_classes):
    fpr_lr[i], tpr_lr[i], _ = roc_curve(y_test_bin[:, i], y_prob_lr[:, i])
    roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])

# --- ROC untuk Decision Tree ---
fpr_dt = dict()
tpr_dt = dict()
roc_auc_dt = dict()
for i in range(n_classes):
    fpr_dt[i], tpr_dt[i], _ = roc_curve(y_test_bin[:, i], y_prob_dt[:, i])
    roc_auc_dt[i] = auc(fpr_dt[i], tpr_dt[i])

# --- Plotting ROC Curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colors = cycle(["aqua", "darkorange", "cornflowerblue"])

# Plot LR
ax1.set_title("ROC Curve - Logistic Regression (OvR)")
for i, color in zip(range(n_classes), colors):
    ax1.plot(
        fpr_lr[i],
        tpr_lr[i],
        color=color,
        lw=2,
        label=f"ROC {target_names[i]} (AUC = {roc_auc_lr[i]:.2f})",
    )
ax1.plot([0, 1], [0, 1], "k--", lw=2)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(loc="lower right")

# Plot DT
ax2.set_title("ROC Curve - Decision Tree (OvR)")
for i, color in zip(range(n_classes), colors):
    ax2.plot(
        fpr_dt[i],
        tpr_dt[i],
        color=color,
        lw=2,
        label=f"ROC {target_names[i]} (AUC = {roc_auc_dt[i]:.2f})",
    )
ax2.plot([0, 1], [0, 1], "k--", lw=2)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right")

plt.show()

# KESIMPULAN PERBANDINGAN
# Mengambil metrik ringkasan (misal: macro avg f1-score)
f1_lr = f1_score(y_test, y_pred_lr, average="macro")
f1_dt = f1_score(y_test, y_pred_dt, average="macro")

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_dt = accuracy_score(y_test, y_pred_dt)

print("\n=============================================")
print("        PERBANDINGAN AKHIR MODEL")
print("=============================================")
print(f"Logistic Regression - Accuracy: {acc_lr:.4f}")
print(f"Decision Tree         - Accuracy: {acc_dt:.4f}")
print(f"\nLogistic Regression - Macro F1-Score: {f1_lr:.4f}")
print(f"Decision Tree         - Macro F1-Score: {f1_dt:.4f}")

if f1_dt > f1_lr:
    print("\nKesimpulan: Decision Tree memiliki performa F1-Score (macro) lebih baik.")
else:
    print(
        "\nKesimpulan: Logistic Regression memiliki performa F1-Score (macro) lebih baik."
    )
