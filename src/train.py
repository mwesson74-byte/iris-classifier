import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

iris = load_iris()
X = iris.data
y = iris.target
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions (first 5):", y_pred[:5])
print("True labels (first 5):", y_test[:5])

accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree accuracy:", accuracy)

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close(fig)

model_path = os.path.join(OUTPUT_DIR, "decision_tree_model.joblib")
joblib.dump(model, model_path)

print(f"Saved confusion matrix to: {cm_path}")
print(f"Saved model to: {model_path}")
