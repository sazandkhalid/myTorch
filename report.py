import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import models as Model

def report(model : Model, X_test, y_test, y_pred = None, *, task = "regression", train_losses=None, val_losses = None, metrics = None, save_path = None,n_samples_to_show=10,):
    if y_pred is None:
        y_pred = model.forward(X_test)
    y_test = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    #Loss Curve 
    if train_losses is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label="Train Loss")
        if val_losses is not None:
            plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        if save_path:
            plt.savefig(f"{save_path}_loss_curve.png", dpi=120)
        plt.show()
    #Parity Plots + Confusion matric for classification tasks 
    if task == "regression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Regression Metrics:\n MSE: {mse:.4f}\n R²: {r2:.4f}")

        # Parity plot
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        min_val, max_val = y_test.min(), y_test.max()
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Parity Plot")
        if save_path:
            plt.savefig(f"{save_path}_parity.png", dpi=120)
        plt.show()

    elif task in ("binary_classification", "multiclass_classification"):
        if task == "binary_classification":
            y_hat = (y_pred >= 0.5).astype(int)
        else:
            y_hat = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_test, y_hat)
        prec = precision_score(y_test, y_hat, average="binary" if task=="binary_classification" else "macro")
        rec = recall_score(y_test, y_hat, average="binary" if task=="binary_classification" else "macro")
        f1 = f1_score(y_test, y_hat, average="binary" if task=="binary_classification" else "macro")
        print(f"Classification Metrics:\n Accuracy: {acc:.3f}\n Precision: {prec:.3f}\n Recall: {rec:.3f}\n F1-score: {f1:.3f}")

        cm = confusion_matrix(y_test, y_hat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        if save_path:
            plt.savefig(f"{save_path}_confusion_matrix.png", dpi=120)
        plt.show()
    #Prediction samples 
    print("\nSample Predictions:")
    for i in range(min(n_samples_to_show, len(y_test))):
        print(f"True: {y_test[i]}, Pred: {y_pred[i]:.4f}")
