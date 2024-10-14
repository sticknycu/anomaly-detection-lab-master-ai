import numpy as np
import pyod.utils.data as pyd
import matplotlib.pyplot as plt
from pyod.models.knn import KNN

from sklearn.metrics import confusion_matrix, roc_curve, auc

X_train, X_test, Y_train, Y_test = pyd.generate_data(n_train=400, n_test=100, contamination=0.1)



print(f'{X_train[:][1]}')
print(f'{X_test.shape}')
print(f'{Y_train.shape}')
print(f'{Y_test.shape}')

for idx, content in enumerate(X_train):
    if Y_train[idx] == 0:
        plt.scatter(content[0], content[1], color='blue')
    if Y_train[idx] == 1:
        plt.scatter(content[0], content[1], color='red')

    # print(idx, content[0], content[1], Y_train[idx])

plt.show()

# ex 2


model = KNN(contamination=0.1)

model.fit(X_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)

tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

def tpr(tp, fn):
    return tp/(tp+fn)

def tnr(tn, fp):
    return tn/(tn+fp)

def balanced_accuracy(tpr, tnr):
    return (tpr + tnr) / 2

print(f'Balanced accuracy: {balanced_accuracy(tpr=tpr(tp, fn), tnr=tnr(tn, fp))}')

x_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, x_pred_proba)

roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Classification')
plt.legend()
plt.show()

# ex 3

X_train, X_test, Y_train, Y_test = pyd.generate_data(n_train=1000, n_test=0, contamination=0.1)

q_50 = np.quantile(X_train, 0.5)

import scipy.stats as stats

z_scores = stats.zscore(X_train)

print(z_scores)

threshold = 2

outliers = np.where(np.abs(z_scores) > threshold)

anomalies = X_train[outliers]

y_pred_anomalies = np.zeros(len(Y_train))
y_pred_anomalies[outliers[0]] = 1  # marcheaza anomaliile cu 1

cm_anomalies = confusion_matrix(Y_train, y_pred_anomalies)

tn_anomalies, fp_anomalies, fn_anomalies, tp_anomalies = cm_anomalies[0, 0], cm_anomalies[0, 1], cm_anomalies[1, 0], cm_anomalies[1, 1]

def tpr(tp, fn):
    return tp/(tp+fn)

def tnr(tn, fp):
    return tn/(tn+fp)

def balanced_accuracy(tpr, tnr):
    return (tpr + tnr) / 2

tpr_anomalies = tpr(tp_anomalies, fn_anomalies)
tnr_anomalies = tnr(tn_anomalies, fp_anomalies)
balanced_acc_anomalies = balanced_accuracy(tpr_anomalies, tnr_anomalies)

print(f'Balanced Accuracy for Anomalies: {balanced_acc_anomalies:.4f}')
