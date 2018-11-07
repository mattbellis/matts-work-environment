import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score


# Getting some of this from here
# https://betatim.github.io/posts/sklearn-for-TMVA-users/

################################################################################
def gen_data(means, sigmas, corrmat, num=100):

    covmat = corrmat
    nsigmas = len(sigmas)
    for i in range(nsigmas):
        for j in range(nsigmas):
            covmat[i][j] *= sigmas[i]*sigmas[j]
    
    dataset = np.random.multivariate_normal(means,covmat,num)

    return dataset
################################################################################

################################################################################
def plot_corr_matrix(ccmat,labels):
    fig1 = plt.figure()
    ax1 = plt.subplot(1,1,1)
    opts = {'cmap': plt.get_cmap("RdBu"), 'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(ccmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)
    ax1.set_title("Correlations")

    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, minor=False)

    plt.tight_layout()
    
    return fig1,ax1
################################################################################


param_labels = ["A", "B", "C", "D"]

################################################################################
# Signal
################################################################################
ndata = 10000
means = [1.0, 1.1, 0.8, 1.1]
sigmas = [0.1, 0.15, 0.2, 0.3]
corrmat = [[1.0, 0.2, 0.5, 0.0], 
           [0.2, 1.0, 0.0, 0.0], 
           [0.5, 0.0, 1.0, 0.4], 
           [0.0, 0.0, 0.4, 1.0]]
data0 = gen_data(means, sigmas, corrmat, ndata)
data0T = data0.transpose()

print(data0)

corrcoefs = []

plt.figure()
for i in range(len(sigmas)):
    corrcoefs.append([])
    for j in range(len(sigmas)):
        plt.subplot(4,4,1+i+4*j)
        x = data0T[i]
        y = data0T[j]
        plt.plot(x,y,'.',markersize=1)
        plt.xlim(0,2)
        plt.ylim(0,2)

        corrcoefs[i].append(np.corrcoef(x,y)[0][1])

print(corrcoefs)

fig,ax = plot_corr_matrix(corrcoefs,param_labels)



################################################################################
# Background
################################################################################
ndata = 10000
means = [1.0, 1.0, 1.0, 1.0]
sigmas = [0.1, 0.05, 0.2, 0.3]
corrmat = [[1.0, 0.0, 0.0, 0.0], 
           [0.0, 1.0, 0.0, 0.0], 
           [0.0, 0.0, 1.0, 0.0], 
           [0.0, 0.0, 0.0, 1.0]]
data1 = gen_data(means, sigmas, corrmat, ndata)
data1T = data1.transpose()

print(data1)

corrcoefs = []

plt.figure()
for i in range(len(sigmas)):
    corrcoefs.append([])
    for j in range(len(sigmas)):
        plt.subplot(4,4,1+i+4*j)
        x = data1T[i]
        y = data1T[j]
        plt.plot(x,y,'.',markersize=1)
        plt.xlim(0,2)
        plt.ylim(0,2)

        corrcoefs[i].append(np.corrcoef(x,y)[0][1])

print(corrcoefs)

fig,ax = plot_corr_matrix(corrcoefs,param_labels)


################################################################################
# Train test split
################################################################################

X = np.concatenate((data0, data1))
y = np.concatenate((np.ones(data0.shape[0]), np.zeros(data1.shape[0])))
print("X -----------------")
print(type(X),X.shape)
print(type(y),y.shape)
print(X)
print(y)

skdataset = {"data":X,"target":y,"target_names":param_labels}

X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size=0.33, random_state=42)
X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev, test_size=0.33, random_state=492)

################################################################################
# Fit/Classify
################################################################################

#dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05*len(X_train))
dt = DecisionTreeClassifier(max_depth=3)

bdt = AdaBoostClassifier(dt, algorithm='SAMME', n_estimators=800, learning_rate=0.5)
bdt.fit(X_train, y_train)

probas = bdt.predict_proba(X_test)
print("probas")
print(probas)
print(y_test)
#plt.figure()
#plt.plot(y_test,probas,'.')

# The order is because we said that the signal was 1 earlier and the background was 0 (labels).
# So those are the columns they get put in for the probabilities.
probbkg,probsig = probas.transpose()[:]
xpts = []
ypts0 = []
ypts1 = []
for i in np.linspace(0,1.0,10000):
    xpts.append(i)
    ypts0.append(len(probsig[(probsig>i)*(y_test==1)]))
    ypts1.append(len(probsig[(probsig>i)*(y_test==0)]))
#print(xpts)
#print(ypts)

n0 = float(len(y_test[y_test==1]))
n1 = float(len(y_test[y_test==0]))
ypts0 = np.array(ypts0)
ypts1 = np.array(ypts1)

plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.plot(xpts,ypts0)
plt.subplot(1,3,2)
plt.plot(xpts,ypts1)
plt.subplot(1,3,3)
plt.plot(ypts1/n1,ypts0/n0)



################################################################################

################################################################################
# Performance
################################################################################
y_pred = bdt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (train) for %s: %0.1f%% " % ("BDT", accuracy * 100))

y_predicted = bdt.predict(X_test)
print(classification_report(y_test, y_predicted, target_names=["background", "signal"]))
print("Area under ROC curve: %.4f"%(roc_auc_score(y_test, bdt.decision_function(X_test))))

y_predicted = bdt.predict(X_train)
print(classification_report(y_train, y_predicted, target_names=["background", "signal"]))
print("Area under ROC curve: %.4f"%(roc_auc_score(y_train, bdt.decision_function(X_train))))


################################################################################
# ROC curve
################################################################################
decisions = bdt.decision_function(X_test)
# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_test, decisions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()

################################################################################
def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30):
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    fig = plt.figure()
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')

    return fig
    
figctt = compare_train_test(bdt, X_train, y_train, X_test, y_test)


plt.show()
