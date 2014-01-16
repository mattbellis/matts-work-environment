import numpy as np
import matplotlib.pyplot as plt
import mlpy
f = np.loadtxt("spiral.data")
x, y = f[:, :2], f[:, 2]
print x
print y
#svm = mlpy.LibSvm(svm_type='c_svc', kernel_type='rbf', gamma=100)
svm = mlpy.LibSvm(svm_type='c_svc', kernel_type='rbf', gamma=100, probability=True)
svm.learn(x, y)
xmin, xmax = x[:,0].min()-0.1, x[:,0].max()+0.1
ymin, ymax = x[:,1].min()-0.1, x[:,1].max()+0.1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
xnew = np.c_[xx.ravel(), yy.ravel()]
#ynew = svm.pred(xnew).reshape(xx.shape)
#ynew = svm.pred_probability(xnew)
ynew = svm.pred_values(xnew)

print ynew

fig = plt.figure(1)
plt.pcolormesh(xx, yy, ynew)
plt.scatter(x[:,0], x[:,1], c=y)
plt.set_cmap(plt.cm.Paired)
plt.show()
