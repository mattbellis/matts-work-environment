import numpy as np
import matplotlib.pylab as plt

import os
import pyhf
import pyhf.readxml
from ipywidgets import interact, fixed

import base64
from IPython.core.display import display, HTML
anim = base64.b64encode(open('workflow.gif','rb').read()).decode('ascii')
HTML('<img src="data:image/gif;base64,{}">'.format(anim))



plt.show()
