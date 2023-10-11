import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

# Завантаження даних з файлу
data = pd.read_csv("bioresponse.csv")

# Розподіл даних на 2 набори: тестовий та навчальний
X = data.drop("Activity", axis=1)
y = data["Activity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення моделей та задання глибини
models = {
    "Small Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Deep Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest with Small Trees": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    "Random Forest with Deep Trees": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Ініціалізація полотна для графіків
plt.figure(figsize=(12, 4))

# Ініціалізація списків для збереження результатів
accuracy_results_before = []
precision_results_before = []
recall_results_before = []
f1_results_before = []
log_loss_results_before = []

accuracy_results_after = []
precision_results_after = []
recall_results_after = []
f1_results_after = []
log_loss_results_after = []

# Побудова Precision-Recall кривих та ROC кривих, обчислення метрик якості
for model_name, model in models.items():

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Розрахунок метрик якості
    accuracy_before = accuracy_score(y_test, (y_pred_proba > 0.5).astype(int))
    precision_before = precision_score(y_test, (y_pred_proba > 0.5).astype(int))
    recall_before = recall_score(y_test, (y_pred_proba > 0.5).astype(int))
    f1_before = f1_score(y_test, (y_pred_proba > 0.5).astype(int))
    loss_before = log_loss(y_test, y_pred_proba)

    accuracy_results_before.append(accuracy_before)
    precision_results_before.append(precision_before)
    recall_results_before.append(recall_before)
    f1_results_before.append(f1_before)
    log_loss_results_before.append(loss_before)

    # Зменшення порогу для уникнення помилок II роду
    threshold = 0.1
    y_pred = (y_pred_proba > threshold).astype(int)

    # Розрахунок метрик якості після уникнення помилок II роду
    accuracy_after = accuracy_score(y_test, y_pred)
    precision_after = precision_score(y_test, y_pred)
    recall_after = recall_score(y_test, y_pred)
    f1_after = f1_score(y_test, y_pred)
    loss_after = log_loss(y_test, y_pred_proba)

    accuracy_results_after.append(accuracy_after)
    precision_results_after.append(precision_after)
    recall_results_after.append(recall_after)
    f1_results_after.append(f1_after)
    log_loss_results_after.append(loss_after)

    # Обчислення точності (precision) і повноти (recall)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    # Побудова графіка Precision-Recall
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f"{model_name}")

    # Обчислення False Positive Rate (FPR) і True Positive Rate (TPR)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # Побудова ROC-кривої
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.', label=f"{model_name} ")

# Відображення графіків ROC-кривої  та Precision-Recall
plt.subplot(1, 2, 1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Before)")
plt.legend(loc="lower left")

plt.subplot(1, 2, 2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Before)")
plt.legend(loc="lower right")

# Виведення інформації про вміст файлу
print("\n Набір даних файлу bioresponse.csv")
print(data.info())
print("=" * 50)


# Виведення результатів обчислених метрик для другого пункту завдання
print("Результати обчислення метрик до уникнення помилок другого роду ")
for model_name, accuracy, precision, recall, f1, log_loss_value in zip(
    models.keys(), accuracy_results_before, precision_results_before, recall_results_before, f1_results_before, log_loss_results_before
):
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {log_loss_value:.4f}")
    print("-" * 50)


# Виведення результатів після уникнення помилок II роду
print("Результати обчислення метрик після уникання помилок II роду:")
for model_name, accuracy, precision, recall, f1, log_loss_value in zip(
    models.keys(), accuracy_results_after, precision_results_after, recall_results_after, f1_results_after, log_loss_results_after
):
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {log_loss_value:.4f}")
    print("=" * 50)



plt.tight_layout()
plt.show()

