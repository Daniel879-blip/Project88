
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from utils.dataset_loader import load_dataset

def get_metrics(model):
    X, y = load_dataset(split='val')
    preds = []
    with torch.no_grad():
        for x in X:
            output = model(x.unsqueeze(0))
            preds.append(output.argmax(1).item())
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    return acc, cm

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
