import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

#sns.set_style('white')

data = np.random.normal(70,10,100)

plt.hist(data,range=(40,100),bins=12)

plt.xlabel('Test scores',fontsize=18)
plt.grid()

plt.savefig('histo_for_quiz.png')

plt.show()
