import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd

import numpy as np

import sys

from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9


df = pd.read_csv(sys.argv[1],header=None,usecols=[0,1,2],names=['x_col','y_col','z_col'], delim_whitespace=True)
#df = np.loadtxt(sys.argv[1],dtype=float,unpack=True)

cvs = ds.Canvas(plot_width=800, plot_height=800)
#agg = cvs.points(df, 'x_col', 'y_col', ds.mean('z_col'))
agg = cvs.points(df, 'x_col', 'y_col', ds.mean('z_col'))
#agg = cvs.points(df, df[0], df[1], ds.mean(df[2]))
img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='log')

background = 'black'

export = partial(export_image, background = background, export_path="export")
cm = partial(colormap_select, reverse=(background!="black"))

export(tf.shade(agg, cmap = cm(Greys9), how='linear'),"testing")

