import pickle
from scipy.special import softmax
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from collections import Counter
from tqdm import tqdm

pickle_location = 'ast-finetuned-audioset-10-10-0.4593-bs8-lr1e-05/checkpoint-24000/logits_labels.pkl'

# Open the file in binary read mode
with open(pickle_location, 'rb') as file:
    # Load the data using pickle
    data = pickle.load(file)

logits = data['logits'][0]
labels = data['labels'][0]

prob = softmax(logits, axis=-1)
pred = np.argmax(logits, axis=-1)

print('MultiClass:')
precision = precision_score(labels, pred, average='macro', zero_division=1)
print(f"Precision: {precision}")
recall = recall_score(labels, pred, average='macro')
print(f"Recall: {recall}")
f1 = f1_score(labels, pred, average='macro')
print(f"F1 Score: {f1}")

roc_auc = roc_auc_score(labels, prob, average='macro', multi_class='ovr')

print(f"ROC AUC: {roc_auc}")

print()
print('MultiLabel:')


labels_onehot = label_binarize(labels, classes=np.arange(logits.shape[1]))
thres = 0.5
_prob = prob
prob = (prob > thres).astype(int)

# Multi-label metrics
precision = precision_score(labels_onehot, prob, average='macro', zero_division=1)
print(f"Precision: {precision}")
recall = recall_score(labels_onehot, prob, average='macro')
print(f"Recall: {recall}")
f1 = f1_score(labels_onehot, prob, average='macro')
print(f"F1 Score: {f1}")
roc_auc = roc_auc_score(labels_onehot, _prob, average='macro', multi_class='ovr')
print(f"ROC AUC: {roc_auc}")