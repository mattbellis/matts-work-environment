
import random

from ROOT import *

from math import sqrt

from math import exp

from math import pi

from math import cos

from math import sin

from math import fabs

#from os import *

import numpy

from array import array

from numpy import matrix

from numpy import float32

from scipy import mat

from scipy import linalg

from glob import glob

import re

import subprocess

import shlex

import sys

#USE_COLOR = True
USE_COLOR = False

################################################################################
# Change this if I want to fit to nominal
################################################################################
FIT_TO_NOMINAL = False
#FIT_TO_NOMINAL = True
#USENOMSTATONLY = True
USENOMSTATONLY = False

################################################################################
# False if I want to calculate my own matrix
################################################################################
FIT_TO_DIRECT_MATRIX = True
#FIT_TO_DIRECT_MATRIX = False

if FIT_TO_NOMINAL and FIT_TO_DIRECT_MATRIX:
    print "ERROR: INCOMPATIBLE FIT FLAGS"
    quit()

################################################################################
# For the covariance matrix printing.
################################################################################
PRINT_FULL_PRECISION = True

#NUMREPITITIONS = 500
#NUMREPITITIONS = 30
NUMREPITITIONS = 10
#NUMREPITITIONS = 1

CONSTRAIN_UPP = True
##### Original value ########
#G_EPSILON = 0.0001
##### Matt Bellis change ######
G_EPSILON = 0.0001

RANDSEED = 100

CALLLIMIT = 100000
#TOLERANCE = 0.001
TOLERANCE = 0.000001

random.seed(RANDSEED)

#if len(sys.argv) < 2:
#  print "Usage: python "+str(sys.argv[0])+" < cov mat log file >"
#  quit()

#if len(sys.argv) < 3:
#    print "Usage: python "+str(sys.argv[0])+" < cov mat log file > <alpha in deg> [random seed]"
#    quit()

if len(sys.argv) < 2:
    print "Usage: python "+str(sys.argv[0])+" < cov mat log file >"
    quit()


inputLogFileName = sys.argv[1]

isChiSq = True

finalPars = []
finalParsErr = []

#alphaScanVal = 120*pi/180.0

alphaInDeg = 0
alphaScanVal = alphaInDeg*pi/180.0



UIVars = ["UPm",
          "UMm",
          "UMp",
          "U0m",
          "U0p",
          "UPMpRe",
          "UPMmRe",
          "UP0pRe",
          "UP0mRe",
          "UM0pRe",
          "UM0mRe",
          "UPMpIm",
          "UPMmIm",
          "UP0pIm",
          "UP0mIm",
          "UM0pIm",
          "UM0mIm",
          "IP",
          "IM",
          "I0",
          "IPMRe",
          "IP0Re",
          "IM0Re",
          "IPMIm",
          "IP0Im",
          "IM0Im"]



humanReadableDict = {}
humanReadableDict["UPm"]    =  "U^{-}_{+}"        # 1
humanReadableDict["UMm"]    =  "U^{-}_{-}"        # 2
humanReadableDict["UMp"]    =  "U^{+}_{-}"        # 3
humanReadableDict["U0m"]    =  "U^{-}_{0}"        # 4
humanReadableDict["U0p"]    =  "U^{+}_{0}"        # 5
humanReadableDict["UPMpRe"] =  "U^{+Re}_{+-}"     # 6
humanReadableDict["UPMmRe"] =  "U^{-Re}_{+-}"     # 7
humanReadableDict["UP0pRe"] =  "U^{+Re}_{+0}"     # 8
humanReadableDict["UP0mRe"] =  "U^{-Re}_{+0}"     # 9
humanReadableDict["UM0pRe"] =  "U^{+Re}_{-0}"     # 10
humanReadableDict["UM0mRe"] =  "U^{-Re}_{-0}"     # 11
humanReadableDict["UPMpIm"] =  "U^{+Im}_{+-}"     # 12
humanReadableDict["UPMmIm"] =  "U^{-Im}_{+-}"     # 13
humanReadableDict["UP0pIm"] =  "U^{+Im}_{+0}"     # 14
humanReadableDict["UP0mIm"] =  "U^{-Im}_{+0}"     # 15
humanReadableDict["UM0pIm"] =  "U^{+Im}_{-0}"     # 16
humanReadableDict["UM0mIm"] =  "U^{-Im}_{-0}"     # 17
humanReadableDict["IP"]     =  "I_{+}"            # 18
humanReadableDict["IM"]     =  "I_{-}"            # 19
humanReadableDict["I0"]     =  "I_{0}"            # 20
humanReadableDict["IPMRe"]  =  "I^{Re}_{+-}"      # 21
humanReadableDict["IP0Re"]  =  "I^{Re}_{+0}"      # 22
humanReadableDict["IM0Re"]  =  "I^{Re}_{-0}"      # 23
humanReadableDict["IPMIm"]  =  "I^{Im}_{+-}"      # 24
humanReadableDict["IP0Im"]  =  "I^{Im}_{+0}"      # 25
humanReadableDict["IM0Im"]  =  "I^{Im}_{-0}"      # 26

inverseHumanReadableDict = {}
inverseHumanReadableDict["N3Pi"]         = "N3Pi"
inverseHumanReadableDict["U^{+}_{+}"]    = "UPp"
inverseHumanReadableDict["U^{-}_{+}"]    = "UPm"
inverseHumanReadableDict["U^{-}_{-}"]    = "UMm"
inverseHumanReadableDict["U^{+}_{-}"]    = "UMp"
inverseHumanReadableDict["U^{-}_{0}"]    = "U0m"
inverseHumanReadableDict["U^{+}_{0}"]    = "U0p"
inverseHumanReadableDict["U^{+Re}_{+-}"] = "UPMpRe"
inverseHumanReadableDict["U^{-Re}_{+-}"] = "UPMmRe"
inverseHumanReadableDict["U^{+Re}_{+0}"] = "UP0pRe"
inverseHumanReadableDict["U^{-Re}_{+0}"] = "UP0mRe"
inverseHumanReadableDict["U^{+Re}_{-0}"] = "UM0pRe"
inverseHumanReadableDict["U^{-Re}_{-0}"] = "UM0mRe"
inverseHumanReadableDict["U^{+Im}_{+-}"] = "UPMpIm"
inverseHumanReadableDict["U^{-Im}_{+-}"] = "UPMmIm"
inverseHumanReadableDict["U^{+Im}_{+0}"] = "UP0pIm"
inverseHumanReadableDict["U^{-Im}_{+0}"] = "UP0mIm"
inverseHumanReadableDict["U^{+Im}_{-0}"] = "UM0pIm"
inverseHumanReadableDict["U^{-Im}_{-0}"] = "UM0mIm"
inverseHumanReadableDict["I_{+}"]        = "IP"
inverseHumanReadableDict["I_{-}"]        = "IM"
inverseHumanReadableDict["I_{0}"]        = "I0"
inverseHumanReadableDict["I^{Re}_{+-}"]  = "IPMRe"
inverseHumanReadableDict["I^{Re}_{+0}"]  = "IP0Re"
inverseHumanReadableDict["I^{Re}_{-0}"]  = "IM0Re"
inverseHumanReadableDict["I^{Im}_{+-}"]  = "IPMIm"
inverseHumanReadableDict["I^{Im}_{+0}"]  = "IP0Im"
inverseHumanReadableDict["I^{Im}_{-0}"]  = "IM0Im"

nominalDict = {}        #Nominal Val StatErr SysErr
nominalDict["U^{+}_{+}"]     = [ 1.00, 0.00, 0.00 ]
nominalDict["U^{+}_{0}"]     = [ 0.28, 0.07, 0.04 ]
nominalDict["U^{+}_{-}"]     = [ 1.32, 0.12, 0.05 ]
nominalDict["U^{-}_{0}"]     = [-0.03, 0.11, 0.09 ]
nominalDict["U^{-}_{-}"]     = [-0.32, 0.14, 0.05 ]
nominalDict["U^{-}_{+}"]     = [ 0.54, 0.15, 0.05 ]

nominalDict["I_{0}"]         = [ 0.01, 0.06, 0.01 ]
nominalDict["I_{-}"]         = [-0.01, 0.10, 0.02 ]
nominalDict["I_{+}"]         = [-0.02, 0.10, 0.03 ]

nominalDict["U^{+Im}_{+-}"]  = [-0.07, 0.71, 0.73]
nominalDict["U^{+Re}_{+-}"]  = [ 0.17, 0.49, 0.31]
nominalDict["U^{-Im}_{+-}"]  = [-0.38, 1.06, 0.36]
nominalDict["U^{-Re}_{+-}"]  = [ 2.23, 1.00, 0.43]

nominalDict["I^{Im}_{+-}"]   = [-1.99, 1.25, 0.34]
nominalDict["I^{Re}_{+-}"]   = [ 1.90, 2.03, 0.65]

nominalDict["U^{+Im}_{+0}"]  = [-0.16, 0.57, 0.14]
nominalDict["U^{+Re}_{+0}"]  = [-1.08, 0.48, 0.20]

nominalDict["U^{-Im}_{+0}"]  = [-1.66, 0.94, 0.25]
nominalDict["U^{-Re}_{+0}"]  = [-0.18, 0.88, 0.35]

nominalDict["I^{Im}_{+0}"]   = [-0.21, 1.06, 0.25]
nominalDict["I^{Re}_{+0}"]   = [ 0.41, 1.30, 0.41]

nominalDict["U^{+Im}_{-0}"]  = [-0.17, 0.50, 0.23]
nominalDict["U^{+Re}_{-0}"]  = [-0.36, 0.38, 0.08]

nominalDict["U^{-Im}_{-0}"]  = [ 0.12, 0.75, 0.22]
nominalDict["U^{-Re}_{-0}"]  = [-0.63, 0.72, 0.32]

nominalDict["I^{Im}_{-0}"]   = [ 1.23, 1.07, 0.29]
nominalDict["I^{Re}_{-0}"]   = [ 0.41, 1.30, 0.21]

nominalNonHumanReadableDict = {}
nominalNonHumanReadableDict["UPp"]     = nominalDict["U^{+}_{+}"]
nominalNonHumanReadableDict["UPm"]     = nominalDict["U^{-}_{+}"]
nominalNonHumanReadableDict["UMm"]     = nominalDict["U^{-}_{-}"]
nominalNonHumanReadableDict["UMp"]     = nominalDict["U^{+}_{-}"]
nominalNonHumanReadableDict["U0m"]     = nominalDict["U^{-}_{0}"]
nominalNonHumanReadableDict["U0p"]     = nominalDict["U^{+}_{0}"]
nominalNonHumanReadableDict["UPMpRe"]  = nominalDict["U^{+Re}_{+-}"]
nominalNonHumanReadableDict["UPMmRe"]  = nominalDict["U^{-Re}_{+-}"]
nominalNonHumanReadableDict["UP0pRe"]  = nominalDict["U^{+Re}_{+0}"]
nominalNonHumanReadableDict["UP0mRe"]  = nominalDict["U^{-Re}_{+0}"]
nominalNonHumanReadableDict["UM0pRe"]  = nominalDict["U^{+Re}_{-0}"]
nominalNonHumanReadableDict["UM0mRe"]  = nominalDict["U^{-Re}_{-0}"]
nominalNonHumanReadableDict["UPMpIm"]  = nominalDict["U^{+Im}_{+-}"]
nominalNonHumanReadableDict["UPMmIm"]  = nominalDict["U^{-Im}_{+-}"]
nominalNonHumanReadableDict["UP0pIm"]  = nominalDict["U^{+Im}_{+0}"]
nominalNonHumanReadableDict["UP0mIm"]  = nominalDict["U^{-Im}_{+0}"]
nominalNonHumanReadableDict["UM0pIm"]  = nominalDict["U^{+Im}_{-0}"]
nominalNonHumanReadableDict["UM0mIm"]  = nominalDict["U^{-Im}_{-0}"]
nominalNonHumanReadableDict["IP"]      = nominalDict["I_{+}"]
nominalNonHumanReadableDict["IM"]      = nominalDict["I_{-}"]
nominalNonHumanReadableDict["I0"]      = nominalDict["I_{0}"]
nominalNonHumanReadableDict["IPMRe"]   = nominalDict["I^{Re}_{+-}"]
nominalNonHumanReadableDict["IP0Re"]   = nominalDict["I^{Re}_{+0}"]
nominalNonHumanReadableDict["IM0Re"]   = nominalDict["I^{Re}_{-0}"]
nominalNonHumanReadableDict["IPMIm"]   = nominalDict["I^{Im}_{+-}"]
nominalNonHumanReadableDict["IP0Im"]   = nominalDict["I^{Im}_{+0}"]
nominalNonHumanReadableDict["IM0Im"]   = nominalDict["I^{Im}_{-0}"]

nominalCorMatrixVars = ["N3Pi",
                        "I_{0}",
                        "I_{-}",
                        "I^{Im}_{-0}",
                        "I^{Re}_{-0}",
                        "I_{+}",
                        "I^{Im}_{+0}",
                        "I^{Re}_{+0}",
                        "I^{Im}_{+-}",
                        "I^{Re}_{+-}",
                        "U^{-}_{0}",
                        "U^{+}_{0}",
                        "U^{-Im}_{-0}",
                        "U^{-Re}_{-0}",
                        "U^{+Im}_{-0}",
                        "U^{+Re}_{-0}",
                        "U^{-}_{-}",
                        "U^{+}_{-}",
                        "U^{-Im}_{+0}",
                        "U^{-Re}_{+0}",
                        "U^{+Im}_{+0}",
                        "U^{+Re}_{+0}",
                        "U^{-Im}_{+-}",
                        "U^{-Re}_{+-}",
                        "U^{+Im}_{+-}",
                        "U^{+Re}_{+-}",
                        "U^{-}_{+}"]

nonHumanReadableNominalCorMatrixVars = []
for i in range(0,len(nominalCorMatrixVars)):
    nonHumanReadableNominalCorMatrixVars.append(inverseHumanReadableDict[nominalCorMatrixVars[i]])

nominalCorrMatrixStatUnbalanced = [[   1.00 ],
                                   [  -0.02  ,   1.00 ],
                                   [  -0.04  ,  -0.04  ,   1.00 ],
                                   [  -0.09  ,  -0.11  ,   0.28  ,   1.00 ],
                                   [  -0.03  ,   0.28  ,  -0.18  ,  -0.15  ,   1.00 ],
                                   [   0.06  ,  -0.04  ,  -0.20  ,  -0.21  ,   0.17  ,   1.00 ],
                                   [   0.06  ,  -0.03  ,  -0.11  ,  -0.14  ,   0.09  ,   0.38  ,   1.00 ],
                                   [  -0.17  ,   0.30  ,   0.18  ,   0.17  ,  -0.06  ,  -0.35  ,  -0.45  ,   1.00 ],
                                   [   0.09  ,   0.11  ,   0.14  ,  -0.17  ,   0.10  ,   0.21  ,   0.11  ,  -0.03  ,   1.00 ],
                                   [  -0.24  ,   0.04  ,   0.36  ,   0.28  ,  -0.15  ,  -0.46  ,  -0.25  ,   0.43  ,  -0.01  ,   1.00 ],
                                   [  -0.03  ,   0.07  ,   0.08  ,  -0.05  ,  -0.11  ,  -0.06  ,  -0.02  ,   0.13  ,   0.09  ,   0.15  ,   1.00 ],
                                   [   0.04  ,  -0.02  ,   0.20  ,   0.32  ,  -0.19  ,  -0.24  ,  -0.19  ,   0.30  ,  -0.13  ,   0.35  ,   0.11  ,   1.00 ],
                                   [   0.01  ,   0.13  ,  -0.11  ,  -0.14  ,   0.42  ,   0.14  ,   0.08  ,  -0.07  ,   0.10  ,  -0.13  ,  -0.36  ,  -0.18  ,   1.00 ],
                                   [   0.02  ,   0.07  ,  -0.05  ,  -0.44  ,   0.03  ,   0.06  ,   0.05  ,   0.01  ,   0.17  ,  -0.06  ,   0.04  ,  -0.19  ,   0.13  ,   1.00 ],
                                   [   0.07  ,   0.18  ,  -0.14  ,  -0.39  ,   0.21  ,   0.22  ,   0.18  ,  -0.14  ,   0.27  ,  -0.26  ,   0.03  ,  -0.56  ,   0.31  ,   0.36  ,   1.00 ],
                                   [  -0.05  ,   0.16  ,   0.07  ,  -0.21  ,   0.07  ,  -0.01  ,  -0.01  ,   0.14  ,   0.25  ,   0.08  ,   0.09  ,  -0.05  ,   0.19  ,   0.34  ,   0.25  ,   1.00 ],
                                   [   0.11  ,  -0.01  ,  -0.03  ,  -0.06  ,   0.01  ,   0.01  ,   0.02  ,  -0.06  ,   0.01  ,  -0.08  ,  -0.09  ,  -0.08  ,   0.12  ,   0.19  ,   0.06  ,   0.03  ,   1.00 ],
                                   [  -0.12  ,   0.14  ,   0.08  ,  -0.02  ,   0.08  ,  -0.06  ,  -0.06  ,   0.22  ,   0.06  ,   0.20  ,   0.11  ,   0.18  ,   0.10  ,   0.11  ,   0.20  ,   0.28  ,  -0.13  ,   1.00 ],
                                   [   0.26  ,   0.03  ,  -0.17  ,  -0.19  ,   0.10  ,   0.17  ,   0.11  ,  -0.20  ,   0.08  ,  -0.40  ,  -0.26  ,  -0.31  ,   0.15  ,   0.07  ,   0.22  ,  -0.04  ,   0.10  ,  -0.16  ,   1.00 ],
                                   [   0.13  ,  -0.02  ,   0.00  ,   0.08  ,  -0.18  ,  -0.13  ,  -0.23  ,   0.07  ,  -0.05  ,  -0.05  ,  -0.04  ,   0.03  ,  -0.11  ,  -0.01  ,  -0.08  ,  -0.05  ,   0.07  ,  -0.05  ,   0.32  ,   1.00 ],
                                   [   0.03  ,   0.12  ,  -0.17  ,  -0.41  ,   0.36  ,   0.34  ,   0.34  ,  -0.31  ,   0.25  ,  -0.29  ,  -0.02  ,  -0.54  ,   0.29  ,   0.22  ,   0.56  ,   0.17  ,  -0.00  ,   0.05  ,   0.20  ,  -0.25  ,   1.00 ],
                                   [   0.23  ,  -0.03  ,  -0.16  ,  -0.25  ,   0.13  ,   0.25  ,   0.28  ,  -0.49  ,   0.16  ,  -0.44  ,  -0.12  ,  -0.36  ,   0.12  ,   0.08  ,   0.23  ,   0.02  ,   0.11  ,  -0.31  ,   0.37  ,  -0.06  ,   0.41  ,   1.00 ],
                                   [  -0.14  ,  -0.04  ,   0.03  ,   0.15  ,  -0.04  ,  -0.08  ,  -0.07  ,   0.09  ,  -0.11  ,   0.19  ,   0.01  ,   0.13  ,  -0.08  ,  -0.15  ,  -0.20  ,  -0.11  ,  -0.27  ,  -0.05  ,  -0.15  ,  -0.03  ,  -0.12  ,  -0.17  ,   1.00 ],
                                   [  -0.12  ,  -0.10  ,  -0.05  ,   0.14  ,  -0.05  ,  -0.01  ,  -0.04  ,  -0.01  ,  -0.20  ,   0.07  ,  -0.10  ,   0.09  ,  -0.04  ,  -0.12  ,  -0.19  ,  -0.21  ,   0.08  ,   0.02  ,  -0.07  ,   0.03  ,  -0.16  ,  -0.16  ,   0.09  ,   1.00 ],
                                   [   0.12  ,  -0.20  ,  -0.09  ,   0.18  ,  -0.21  ,  -0.03  ,  -0.01  ,  -0.17  ,  -0.18  ,  -0.22  ,  -0.17  ,   0.02  ,  -0.17  ,  -0.22  ,  -0.33  ,  -0.35  ,   0.06  ,  -0.48  ,   0.15  ,   0.22  ,  -0.33  ,   0.12  ,   0.10  ,   0.12  ,   1.00 ],
                                   [  -0.15  ,   0.05  ,   0.07  ,   0.01  ,   0.06  ,   0.00  ,  -0.03  ,   0.12  ,   0.22  ,   0.13  ,   0.10  ,   0.10  ,   0.03  ,   0.03  ,  -0.04  ,   0.17  ,  -0.02  ,   0.10  ,  -0.19  ,  -0.19  ,   0.05  ,  -0.07  ,   0.03  ,  -0.15  ,  -0.24  ,   1.00 ],
                                   [  -0.05  ,  -0.01  ,   0.00  ,   0.05  ,  -0.06  ,  -0.07  ,  -0.05  ,   0.05  ,  -0.03  ,   0.04  ,  -0.03  ,   0.05  ,  -0.04  ,  -0.02  ,  -0.03  ,  -0.03  ,  -0.07  ,   0.11  ,   0.14  ,   0.30  ,  -0.11  ,  -0.11  ,   0.18  ,   0.20  ,   0.02  ,  -0.12  ,   1.00 ]]

nominalCorrMatrixSystUnbalanced = [[   1.00 ],
                                   [   0.10  ,   1.00 ],
                                   [   0.07  ,  -0.20  ,   1.00 ],
                                   [   0.00  ,  -0.33  ,   0.78  ,   1.00 ],
                                   [   0.29  ,   0.17  ,  -0.52  ,  -0.56  ,   1.00 ],
                                   [  -0.13  ,   0.53  ,   0.04  ,  -0.18  ,   0.31  ,   1.00 ],
                                   [  -0.07  ,  -0.25  ,   0.58  ,   0.70  ,  -0.25  ,   0.30  ,   1.00 ],
                                   [   0.20  ,   0.36  ,  -0.21  ,  -0.10  ,  -0.10  ,  -0.51  ,  -0.58  ,   1.00 ],
                                   [  -0.03  ,  -0.35  ,  -0.33  ,  -0.14  ,   0.37  ,   0.23  ,   0.31  ,  -0.55  ,   1.00 ],
                                   [   0.36  ,   0.55  ,   0.01  ,   0.13  ,  -0.20  ,  -0.18  ,  -0.18  ,   0.75  ,  -0.54  ,   1.00 ],
                                   [   0.01  ,   0.90  ,  -0.30  ,  -0.39  ,   0.39  ,   0.71  ,  -0.14  ,   0.09  ,  -0.07  ,   0.28  ,   1.00 ],
                                   [  -0.11  ,  -0.75  ,   0.46  ,   0.58  ,  -0.55  ,  -0.71  ,   0.22  ,   0.04  ,  -0.14  ,  -0.12  ,  -0.89  ,   1.00 ],
                                   [   0.49  ,  -0.66  ,   0.29  ,   0.34  ,  -0.11  ,  -0.49  ,   0.29  ,  -0.17  ,   0.25  ,  -0.20  ,  -0.73  ,   0.62  ,   1.00 ],
                                   [  -0.39  ,   0.53  ,  -0.50  ,  -0.55  ,   0.35  ,   0.55  ,  -0.24  ,   0.03  ,   0.25  ,  -0.08  ,   0.71  ,  -0.71  ,  -0.76   ,   1.00 ],
                                   [  -0.07  ,   0.15  ,  -0.40  ,  -0.49  ,  -0.22  ,  -0.31  ,  -0.54  ,   0.50  ,  -0.19  ,   0.17  ,  -0.00  ,  -0.05  ,  -0.11   ,   0.29  ,   1.00 ],
                                   [  -0.31  ,  -0.58  ,   0.26  ,   0.12  ,  -0.04  ,   0.08  ,   0.36  ,  -0.75  ,   0.33  ,  -0.84  ,  -0.39  ,   0.26  ,   0.28   ,  -0.14  ,  -0.29  ,   1.00 ],
                                   [  -0.05  ,  -0.44  ,  -0.25  ,  -0.22  ,   0.15  ,  -0.26  ,  -0.15  ,  -0.10  ,   0.46  ,  -0.34  ,  -0.38  ,   0.24  ,   0.36   ,  -0.04  ,   0.11  ,   0.16  ,   1.00 ],
                                   [  -0.10  ,   0.76  ,  -0.24  ,  -0.35  ,   0.04  ,   0.36  ,  -0.26  ,   0.39  ,  -0.24  ,   0.33  ,   0.74  ,  -0.61  ,  -0.67   ,   0.73  ,   0.51  ,  -0.40  ,  -0.42  ,   1.00 ],
                                   [   0.87  ,   0.07  ,  -0.10  ,  -0.17  ,   0.51  ,  -0.07  ,  -0.16  ,   0.14  ,   0.17  ,   0.10  ,   0.06  ,  -0.20  ,   0.48   ,  -0.17  ,  -0.05  ,  -0.14  ,   0.14  ,  -0.07  ,   1.00 ],
                                   [  -0.18  ,  -0.50  ,  -0.00  ,   0.00  ,  -0.41  ,  -0.74  ,  -0.38  ,   0.35  ,  -0.14  ,  -0.12  ,  -0.62  ,   0.59  ,   0.24   ,  -0.14  ,   0.58  ,   0.15  ,   0.37  ,  -0.09  ,  -0.12  ,   1.00 ],
                                   [  -0.05  ,  -0.50  ,  -0.09  ,  -0.01  ,  -0.19  ,  -0.27  ,   0.16  ,  -0.27  ,   0.18  ,  -0.18  ,  -0.59  ,   0.40  ,   0.52   ,  -0.47  ,   0.14  ,   0.19  ,   0.28  ,  -0.57  ,  -0.13  ,   0.11  ,   1.00 ],
                                   [   0.03  ,  -0.23  ,  -0.06  ,  -0.08  ,   0.52  ,   0.49  ,   0.35  ,  -0.79  ,   0.65  ,  -0.58  ,   0.05  ,  -0.25  ,   0.14   ,   0.04  ,  -0.67  ,   0.53  ,   0.22  ,  -0.46  ,   0.20  ,  -0.53  ,   0.14  ,   1.00 ],
                                   [   0.33  ,   0.20  ,   0.27  ,   0.47  ,  -0.27  ,  -0.12  ,   0.28  ,   0.25  ,  -0.34  ,   0.65  ,  -0.05  ,   0.17  ,   0.24   ,  -0.55  ,  -0.29  ,  -0.46  ,  -0.28  ,  -0.23  ,   0.07  ,  -0.39  ,   0.28  ,  -0.16  ,   1.00 ],
                                   [   0.05  ,   0.62  ,   0.25  ,   0.22  ,  -0.05  ,   0.35  ,   0.14  ,   0.26  ,  -0.43  ,   0.43  ,   0.61  ,  -0.38  ,  -0.55   ,   0.30  ,  -0.10  ,  -0.34  ,  -0.73  ,   0.64  ,  -0.09  ,  -0.36  ,  -0.69  ,  -0.32  ,   0.12  ,   1.00 ],
                                   [  -0.08  ,  -0.84  ,   0.27  ,   0.34  ,  -0.50  ,  -0.69  ,   0.14  ,  -0.09  ,   0.12  ,  -0.33  ,  -0.94  ,   0.86  ,   0.72   ,  -0.60  ,   0.20  ,   0.42  ,   0.42  ,  -0.59  ,  -0.09  ,   0.73  ,   0.58  ,  -0.16  ,  -0.05  ,  -0.62  ,   1.00 ],
                                   [  -0.04  ,   0.62  ,   0.13  ,   0.09  ,   0.21  ,   0.68  ,   0.27  ,  -0.20  ,  -0.22  ,   0.17  ,   0.69  ,  -0.54  ,  -0.49   ,   0.21  ,  -0.53  ,  -0.07  ,  -0.55  ,   0.29  ,  -0.08  ,  -0.81  ,  -0.39  ,   0.28  ,   0.31  ,   0.59  ,  -0.77  ,   1.00 ],
                                   [   0.14  ,   0.39  ,  -0.34  ,  -0.34  ,   0.10  ,  -0.06  ,  -0.40  ,   0.53  ,  -0.05  ,   0.36  ,   0.32  ,  -0.25  ,  -0.15   ,   0.35  ,   0.46  ,  -0.48  ,   0.38  ,   0.45  ,   0.24  ,   0.17  ,  -0.23  ,  -0.36  ,  -0.03  ,  -0.02  ,  -0.21  ,  -0.14  ,   1.00 ]]


def defaultHistoSettings(h):
    h.SetNdivisions(6)
    h.GetYaxis().SetTitleSize(0.09)
    h.GetYaxis().SetTitleFont(42)
    h.GetYaxis().SetTitleOffset(0.7)
    h.GetYaxis().CenterTitle()
    h.GetYaxis().SetNdivisions(6)
#    h.GetYaxis().SetTitle("events/MeV")
    h.SetFillColor(9)

def defaultPadSettings():
    
    gPad.SetFillColor(0)
    gPad.SetBorderSize(0)
    gPad.SetRightMargin(0.20);
    gPad.SetLeftMargin(0.20);
    gPad.SetBottomMargin(0.15);

def waitForInput():
    rep = ''
    while not rep in [ 'c', 'C' ]:
        rep = raw_input( 'enter "c" to continue: ' )
        if 1 < len(rep):
            rep = rep[0]

class bcolors:

    GREY = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m'
    BLUE = '\033[94m'
    VIOLET = '\033[95m'
    AQUA = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'

    def disable(self):

        self.GREY = ''
        self.RED = ''
        self.GREEN = ''
        self.ORANGE = ''
        self.BLUE = ''
        self.VIOLET = ''
        self.AQUA = ''
        self.WHITE = ''
        self.ENDC = ''

numMostCorrelated = 5

min = 100
max = 149

limitsArray = [0.2,
               0.4,
               0.6,
               0.8]

def printSquareMatrix(M,useColor):

    bc = bcolors()
    if not useColor:
        bc.disable()

    if useColor:
        print bc.RED + "Over " + str(limitsArray[3]) + bc.ENDC
        print bc.ORANGE + "Over " + str(limitsArray[2]) + bc.ENDC
        print bc.VIOLET + "Over " + str(limitsArray[1]) + bc.ENDC
        print bc.AQUA + "Over " + str(limitsArray[0]) + bc.ENDC
        print bc.BLUE + "Other" + bc.ENDC

    dim = len(M)

    print "   ",
    for colNum in range(1,dim+1):
        if colNum<dim:
            colString = "  %2d   " % colNum
        else:
            colString = "  %2d\n" % colNum
        print colString,
    print "  " + "-"*207

    for i in range(0,dim):
        row = i+1
        rowString = "%2d|" % row
        print rowString,

        for j in range(0,dim):

            outstring = ''

            val = abs(M[i][j])

            if val >= limitsArray[3]:
                outstring = bc.RED
            elif val >= limitsArray[2]:
                outstring = bc.ORANGE
            elif val >= limitsArray[1]:
                outstring = bc.VIOLET
            elif val >= limitsArray[0]:
                outstring = bc.AQUA
            else:
                outstring = bc.BLUE
            
            if M[i][j] != None:

                tempString = None

                if PRINT_FULL_PRECISION:
                    tempString = "%3.2e" % M[i][j]
                else:
                    tempString = "%3.2f" % M[i][j]

                padding = 5 - len(tempString)
                suffix = " "*padding + tempString
                outstring = outstring + suffix + bc.ENDC
            else:
                outstring = outstring + "None" + bc.ENDC

            if j == dim-1:
                outstring = outstring + "\n"
            else:
                valLength = 0
                if M[i][j] == "None":
                    valLength = 4
                else:
                    valLength = 5
                padding = 7 - valLength
                outstring = outstring + " "*padding

            print outstring,

    print "  " + "-"*207

################################################################################
################################################################################
# This is what is parsing the log file!!!!!!!!!!!!!!
################################################################################
################################################################################
def parseMatrix(inputFileName):
    
    curLogData = [line for line in file(inputFileName)]
    

    
    if len(curLogData) == 0:
        print "Warning:", inputFileName, "is empty"
        quit()


    lastCovMatrixIndex = None
    for i in range(0,len(curLogData)):
        if re.search('EXTERNAL ERROR MATRIX',curLogData[i]) != None:
            splitLine = curLogData[i].strip().split()
            ndim = int(splitLine[4])
            if ndim>26:
                lastCovMatrixIndex = i+2

    endOfCovMatrixIndex = None
    for i in range(lastCovMatrixIndex,len(curLogData)):
        if re.search('PARAMETER  CORRELATION COEFFICIENTS  ',curLogData[i]) != None:
            endOfCovMatrixIndex = i
            break

#    print "GREPCOV:", curLogData[endOfCovMatrixIndex]

#    print "GREPCOV:", curLogData[lastCovMatrixIndex]
#    print "GREPCOV:", curLogData[endOfCovMatrixIndex]

    currentVarNum = 0
    j = lastCovMatrixIndex
    directCovMatrixRaw = []
    while j < endOfCovMatrixIndex :
        numReps=0
        exitCurrentLoop = False
        while not exitCurrentLoop:
            if currentVarNum < (numReps+1)*11:
                exitCurrentLoop = True
            else:
                numReps += 1
        currentLine = curLogData[j].strip().split()
        if numReps>0:
            j = j+numReps
            finalCurrentLine = curLogData[j].strip().split()
            currentLine = currentLine+finalCurrentLine[11:]

        directCovMatrixRaw.append(currentLine)
        currentVarNum += 1
        j += 1
    
#    for curRow in directCovMatrixRaw:
#        joinedRow = " ".join(curRow)
#        print "GREPCOV "+joinedRow
    
    numRawVars = currentVarNum
#    print "GREPNUMRAW", numRawVars, len(directCovMatrixRaw)
 
    directCovMatrixSquare = [[None for j in range(0,numRawVars)] for i in range(0,numRawVars)]
    for i in range(0,len(directCovMatrixRaw)):
        for j in range(0,len(directCovMatrixRaw[i])):
            # ---- Matt's stuff ----- Bellis --------- #
            directCovMatrixSquare[i][j] = float(directCovMatrixRaw[i][j])
            '''
            if float(directCovMatrixRaw[i][j]) < 0.0000001:
                directCovMatrixSquare[i][j] = 0.0001
            '''

    for i in range(0,numRawVars):
        for j in range(0,numRawVars):
            if directCovMatrixSquare[i][j] == None:
                directCovMatrixSquare[i][j] = directCovMatrixSquare[j][i]
            elif directCovMatrixSquare[i][j] != directCovMatrixSquare[j][i]:
                print "ERROR: Non-Symmetric direct covariance matrix"

    

    lastCorrMatrixIndex = None
    
    for i in range(0,len(curLogData)):
        if re.search('NO.  GLOBAL      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15     16',curLogData[i]) != None:
            lastCorrMatrixIndex = i+1
    
    quitProgram = False
    
    if lastCorrMatrixIndex == None:
        print "Warning: Could not find last correlation matrix index"
        quitProgram = True
    
    if quitProgram:
        quit()

    firstVarIndex = None
    lastVarIndex = None
    for i in range(0,len(curLogData)):
        if re.search('var-name  gener',curLogData[i]) != None:
            firstVarIndex = i+1
        if re.search("Macro: 'grep' output",curLogData[i]) != None:
            lastVarIndex = i-1
            break
    numVars = lastVarIndex-firstVarIndex+1
    print "Found", numVars, "variables"

    varNameList = []
    varValList = []
    varErrList = []
    for i in range(firstVarIndex,lastVarIndex+1):
        curVarLine = curLogData[i]
        curVarLineTokenized = curVarLine.strip().split()
        varNameList.append(curVarLineTokenized[0])
        varValList.append(float(curVarLineTokenized[3]))
        varErrList.append(float(curVarLineTokenized[4]))

    for i in range(0,len(varNameList)):
        print i+1,"\t",varNameList[i],"\t= ",varValList[i], "\t+/- ",varErrList[i]


#    print "---------------"
#    for i in range(firstVarIndex,lastVarIndex+1):
#        print curLogData[i]
#    print "---------------"

    curIndex = lastCorrMatrixIndex
    quitLoop = False
    while not quitLoop:
      curLine = curLogData[curIndex]
      if re.search('Macro: fit results',curLine) != None:
        quitLoop = True
      else:
        curIndex += 1
    lastCorrMatrixEndIndex = curIndex - 1
    
    corrMatrixLines = curLogData[lastCorrMatrixIndex:lastCorrMatrixEndIndex]

#    print "---------------"
#    print corrMatrixLines[len(corrMatrixLines)-1]
#    print "---------------"

    for i in range(0,len(corrMatrixLines)):
        corrMatrixLines[i] = corrMatrixLines[i].strip()
    
    #print "\n\n\n\n\nRaw Correlation Matrix:"
    #print "---------------------------------------------------------------------------------"
    #for curLine in corrMatrixLines:
    #    print curLine
    #print "---------------------------------------------------------------------------------"
    
    corrMatrix = [[None for j in range(0,numVars)] for i in range(0,numVars)]
    rowIndex = -1
    colIndex = -1
    lastNewRowIndex = 0


    previousLine = "None"
    for curLineIndex,curLine in enumerate(corrMatrixLines+["1000 0 0 0"]):
        tokenizedLine = curLine.strip().split()

        if(float(tokenizedLine[0])>1 and rowIndex != -1 and curLineIndex-lastNewRowIndex>1):
            for j in range(0,len(previousLine)):
                colIndex += 1
                #print previousLine[j]
                curVal = float(previousLine[j])
                #print "col=", colIndex
                corrMatrix[rowIndex][colIndex] = curVal

        if((float(tokenizedLine[0])>1 or rowIndex == -1) and float(tokenizedLine[0]) != 1000):

            #print "(",tokenizedLine[0],")"
            rowIndex += 1
            colIndex = -1

            lastNewRowIndex = curLineIndex

            tokenizedLine = tokenizedLine[2:]

            for j in range(0,len(tokenizedLine)):
                colIndex += 1

                #print "col=", colIndex

                curVal = float(tokenizedLine[j])
                corrMatrix[rowIndex][colIndex] = curVal
            
        previousLine = tokenizedLine

    
    #print "\n\n\n\n\nUnbalanced Correlation Matrix:"
    #print "---------------------------------------------------------------------------------"
    #printSquareMatrix(corrMatrix)
    #print "---------------------------------------------------------------------------------"
    
    for i in range(0,numVars):
        for j in range(0,numVars):
            if corrMatrix[i][j] == None:
                corrMatrix[i][j] = corrMatrix[j][i]
            elif corrMatrix[i][j] != corrMatrix[j][i]:
                print "ERROR: Non-symmetric matrix:", corrMatrix[i][j], corrMatrix[j][i]
    
#    if PRINT_MATRIX:
#    
#        print bc.RED + "Over " + str(limitsArray[3]) + bc.ENDC
#        print bc.ORANGE + "Over " + str(limitsArray[2]) + bc.ENDC
#        print bc.VIOLET + "Over " + str(limitsArray[1]) + bc.ENDC
#        print bc.AQUA + "Over " + str(limitsArray[0]) + bc.ENDC
#        print bc.BLUE + "Other" + bc.ENDC
#    
#        print "Correlation Matrix:"
#        printSquareMatrix(corrMatrix)
    
    
    correlationTuples = []
    for i in range(0,numVars):
        for j in range(0,i):
            tuple = [i,j,corrMatrix[i][j]]
            correlationTuples.append(tuple)
    
    sortedCorrelationTuples = sorted(correlationTuples, key=lambda tuple: abs(tuple[2]), reverse=True)
        
    mostCorrelated = sortedCorrelationTuples[:numMostCorrelated]
    
#    print "             row  col   Corr"
#    for tuple in mostCorrelated:
#        row = tuple[0]+1
#        col = tuple[1]+1
#        val = tuple[2]
#        outstring = "CORRELATION: %2d   %2d    %3.2f" % (row, col, val)
#        print outstring
    
    #numpyCorrMatrix = numpy.array(corrMatrix)


    covMatrix = [[None for j in range(0,numVars)] for i in range(0,numVars)]
    print "numVars: %d" % (numVars)
    for i in range(0,numVars):
        for j in range(0,numVars):
            covMatrix[i][j] = corrMatrix[i][j] * (varErrList[i]*varErrList[j])
            ##### Bellis edit ######
            #### Trying old correlation matrix ######
            #covMatrix[i][j] = nominalUsIsCorrMatrixStat[i][j] * (varErrList[i]*varErrList[j])

    return [corrMatrix, mostCorrelated, varNameList, varValList, varErrList, covMatrix, directCovMatrixSquare]






#############################################################################################################################################################################################
#############################################################################################################################################################################################
#############################################################################################################################################################################################



numNomCorVars = len(nominalCorrMatrixStatUnbalanced)
nominalCorrMatrixStatBalanced = []
nominalCorrMatrixSystBalanced = []
for curRowNum in range(0,numNomCorVars):
    curRow = nominalCorrMatrixStatUnbalanced[curRowNum]
    neededSlots = numNomCorVars-len(curRow)
    tempRow = curRow + [None]*neededSlots
    nominalCorrMatrixStatBalanced.append(tempRow)
for curRowNum in range(0,numNomCorVars):
    curRow = nominalCorrMatrixSystUnbalanced[curRowNum]
    neededSlots = numNomCorVars-len(curRow)
    tempRow = curRow + [None]*neededSlots
    nominalCorrMatrixSystBalanced.append(tempRow)

for i in range(0,numNomCorVars):
    for j in range(0,numNomCorVars):
        curStatVal = nominalCorrMatrixStatBalanced[i][j]
        curSystVal = nominalCorrMatrixSystBalanced[i][j]

        if curStatVal == None:
            nominalCorrMatrixStatBalanced[i][j] = nominalCorrMatrixStatBalanced[j][i]
        if curSystVal == None:
            nominalCorrMatrixSystBalanced[i][j] = nominalCorrMatrixSystBalanced[j][i]

nominalUIVarMasterIndex = {}
for i in range(0,len(nonHumanReadableNominalCorMatrixVars)):
    curName = nonHumanReadableNominalCorMatrixVars[i]
    if curName in UIVars:
        nominalUIVarMasterIndex[curName] = i

numUsIs = len(UIVars)        
nominalUsIsCorrMatrixStat = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
nominalUsIsCorrMatrixSyst = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
nominalUsIsCovMatrixStat =  [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
nominalUsIsCovMatrixSyst =  [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
for i in range(0,numUsIs):
    for j in range(0,numUsIs):
        curIVar = UIVars[i]
        curJVar = UIVars[j]

        curIVarIndex = nominalUIVarMasterIndex[curIVar]
        curJVarIndex = nominalUIVarMasterIndex[curJVar]

        curIVarStatErr = nominalNonHumanReadableDict[curIVar][1]
        curJVarStatErr = nominalNonHumanReadableDict[curJVar][1]

        curIVarSystErr = nominalNonHumanReadableDict[curIVar][2]
        curJVarSystErr = nominalNonHumanReadableDict[curJVar][2]

        nominalUsIsCorrMatrixStat[i][j] = nominalCorrMatrixStatBalanced[curIVarIndex][curJVarIndex]
        nominalUsIsCorrMatrixSyst[i][j] = nominalCorrMatrixSystBalanced[curIVarIndex][curJVarIndex]

        nominalUsIsCovMatrixStat[i][j] = curIVarStatErr*curJVarStatErr*nominalCorrMatrixStatBalanced[curIVarIndex][curJVarIndex]
        nominalUsIsCovMatrixSyst[i][j] = curIVarSystErr*curJVarSystErr*nominalCorrMatrixSystBalanced[curIVarIndex][curJVarIndex]



#for i in range(0,numUsIs):
#    for j in range(0,numUsIs):
#        if i!=j:
#            nominalUsIsCovMatrixStat[i][j] = 0
#            nominalUsIsCovMatrixSyst[i][j] = 0




print "---------------------------------------------------------"
print "Full Balanced Nominal Correlation Matrix (stat):"
print "---------------------------------------------------------"
printSquareMatrix(nominalCorrMatrixStatBalanced,USE_COLOR)

print "---------------------------------------------------------"
print "Full Balanced Nominal Correlation Matrix (syst):"
print "---------------------------------------------------------"
printSquareMatrix(nominalCorrMatrixSystBalanced,USE_COLOR)


print "---------------------------------------------------------"
print "Nominal Correlation Matrix (stat):"
print "---------------------------------------------------------"
printSquareMatrix(nominalUsIsCorrMatrixStat,USE_COLOR)

print "---------------------------------------------------------"
print "Nominal Correlation Matrix (syst):"
print "---------------------------------------------------------"
printSquareMatrix(nominalUsIsCorrMatrixSyst,USE_COLOR)


print "---------------------------------------------------------"
print "Nominal Covariance Matrix (stat):"
print "---------------------------------------------------------"
printSquareMatrix(nominalUsIsCovMatrixStat,USE_COLOR)

print "---------------------------------------------------------"
print "Nominal Covariance Matrix (syst):"
print "---------------------------------------------------------"
printSquareMatrix(nominalUsIsCovMatrixSyst,USE_COLOR)


nominalUIVarVals = []
for i in range(0,numUsIs):
    nominalUIVarVals.append(nominalNonHumanReadableDict[UIVars[i]][0])
    print ' ------ Matts debug UIVars fill Nominal------- '
    print UIVars[i]
    print nominalNonHumanReadableDict[UIVars[i]][0]
    print ' ------ Matts debug UIVars fill ------- '


numpyNomStatCovMatrix = numpy.array(nominalUsIsCovMatrixStat)
scipyNomStatCovMatrix = mat(numpyNomStatCovMatrix)
scipyNomStatInvCovMatrix = linalg.inv(scipyNomStatCovMatrix)

numpyNomSystCovMatrix = numpy.array(nominalUsIsCovMatrixSyst)
scipyNomSystCovMatrix = mat(numpyNomSystCovMatrix)
scipyNomSystInvCovMatrix = linalg.inv(scipyNomSystCovMatrix)

totalNominalCovMatrix = None
if USENOMSTATONLY:
    totalNominalCovMatrix = scipyNomStatCovMatrix
else:
    totalNominalCovMatrix = scipyNomStatCovMatrix + scipyNomSystCovMatrix
scipyTotNomInvCovMatrix = linalg.inv(totalNominalCovMatrix)


NominalUsIsInvCovMatrixStat = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
for i in range(0,numUsIs):
    for j in range(0,numUsIs):
        NominalUsIsInvCovMatrixStat[i][j] = scipyNomStatInvCovMatrix[i,j]
NominalUsIsInvCovMatrixSyst = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
for i in range(0,numUsIs):
    for j in range(0,numUsIs):
        NominalUsIsInvCovMatrixSyst[i][j] = scipyNomSystInvCovMatrix[i,j]
NominalUsIsInvCovMatrixTot = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
for i in range(0,numUsIs):
    for j in range(0,numUsIs):
        NominalUsIsInvCovMatrixTot[i][j] = scipyTotNomInvCovMatrix[i,j]

print "---------------------------------------------------------"
print "Nominal UsIs Inverse Covariance Matrix (stat):"
print "---------------------------------------------------------"
printSquareMatrix(NominalUsIsInvCovMatrixStat,False)

print "---------------------------------------------------------"
print "Nominal UsIs Inverse Covariance Matrix (syst):"
print "---------------------------------------------------------"
printSquareMatrix(NominalUsIsInvCovMatrixSyst,False)

print "---------------------------------------------------------"
print "Nominal UsIs Inverse Covariance Matrix (total):"
print "---------------------------------------------------------"
printSquareMatrix(NominalUsIsInvCovMatrixTot,False)




# To scan using nominal values, set UIVarVals = nominalUIVarVals
# and scipyInvCovMatrix = scipyTotNomInvCovMatrix





#############################################################################################################################################################################################
#############################################################################################################################################################################################
#############################################################################################################################################################################################


parsedTuple = parseMatrix(inputLogFileName)
correlationMatrix = parsedTuple[0]
mostCorrelatedList = parsedTuple[1]

varNames = parsedTuple[2]
varVals = parsedTuple[3]
varErrs = parsedTuple[4]
covarianceMatrix = parsedTuple[5]
directCovarianceMatrix = parsedTuple[6]

print "---------------------------------------------------------"
print "Correlation Matrix:"
print "---------------------------------------------------------"
printSquareMatrix(correlationMatrix,USE_COLOR)

print "---------------------------------------------------------"
print "Covariance Matrix:"
print "---------------------------------------------------------"
printSquareMatrix(covarianceMatrix,False)

print "---------------------------------------------------------"
print "Direct Covariance Matrix:"
print "---------------------------------------------------------"
printSquareMatrix(directCovarianceMatrix,False)

UIVarMasterIndex = {}
for i in range(0,len(varNames)):
    curName = varNames[i]
    if curName in UIVars:
        UIVarMasterIndex[curName] = i

numUsIs = len(UIVars)
UsIsCovMatrix = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
UsIsCorMatrix = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
UsIsDirectCovMatrix = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
for i in range(0,numUsIs):
    for j in range(0,numUsIs):
        curIVar = UIVars[i]
        curJVar = UIVars[j]

        curIVarIndex = UIVarMasterIndex[curIVar]
        curJVarIndex = UIVarMasterIndex[curJVar]

        UsIsCorMatrix[i][j] = correlationMatrix[curIVarIndex][curJVarIndex]
        UsIsCovMatrix[i][j] = covarianceMatrix[curIVarIndex][curJVarIndex]
        UsIsDirectCovMatrix[i][j] = directCovarianceMatrix[curIVarIndex][curJVarIndex]

UIVarVals = []
for i in range(0,numUsIs):
    UIVarVals.append(varVals[UIVarMasterIndex[UIVars[i]]])
    print ' ------ Matts debug UIVars fill standard ------- '
    print UIVars[i]
    print UIVarMasterIndex[UIVars[i]]
    print varVals[UIVarMasterIndex[UIVars[i]]]
    print ' ------ Matts debug UIVars fill ------- '

print "---------------------------------------------------------"
print "UsIs Correlation Matrix:"
print "---------------------------------------------------------"
printSquareMatrix(UsIsCorMatrix,USE_COLOR)

print "---------------------------------------------------------"
print "UsIs Covariance Matrix:"
print "---------------------------------------------------------"
printSquareMatrix(UsIsCovMatrix,False)

print "---------------------------------------------------------"
print "UsIs Direct Covariance Matrix:"
print "---------------------------------------------------------"
printSquareMatrix(UsIsDirectCovMatrix,False)

################################################################################
# Try scaling the cov matrix?  Bellis
################################################################################
'''
print " ---------- Matt's debug ---------- Bellis "
for i in range(0,26):
    for j in range(0,26):
        num = -999.99999999999999
        if UsIsCorMatrix[i][j]!=0.0 and nominalUsIsCorrMatrixStat[i][j]!=0:
            num = nominalUsIsCorrMatrixStat[i][j]/UsIsCorMatrix[i][j]
            UsIsDirectCovMatrix[i][j] /= num
'''
################################################################################


numpyCovMatrix = numpy.array(UsIsCovMatrix)
scipyCovMatrix = mat(numpyCovMatrix)

numpyDirectCovMatrix = numpy.array(UsIsDirectCovMatrix)
scipyDirectCovMatrix = mat(numpyDirectCovMatrix)

scipyInvCovMatrix = linalg.inv(scipyCovMatrix)
scipyDirectInvCovMatrix = linalg.inv(scipyDirectCovMatrix)

#print scipyCovMatrix
#print scipyInvCovMatrix

numUsIs = len(UIVars)
UsIsInvCovMatrix = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
UsIsDirectInvCovMatrix = [[None for j in range(0,numUsIs)] for i in range(0,numUsIs)]
for i in range(0,numUsIs):
    for j in range(0,numUsIs):
        UsIsInvCovMatrix[i][j] = scipyInvCovMatrix[i,j]
        UsIsDirectInvCovMatrix[i][j] = scipyDirectInvCovMatrix[i,j]
#        UsIsInvCovMatrix[i][j] = scipyInvCovMatrix[i][j]

print "---------------------------------------------------------"
print "UsIs Inverse Covariance Matrix:"
print "---------------------------------------------------------"
printSquareMatrix(UsIsInvCovMatrix,False)

print "---------------------------------------------------------"
print "UsIs Direct Inverse Covariance Matrix:"
print "---------------------------------------------------------"
printSquareMatrix(UsIsDirectInvCovMatrix,False)


if FIT_TO_NOMINAL:
    UIVarVals = nominalUIVarVals
    scipyInvCovMatrix = scipyTotNomInvCovMatrix
##    scipyInvCovMatrix = scipyNomStatInvCovMatrix
##    scipyInvCovMatrix = scipyNomSystInvCovMatrix

    #trueParams = [3.34483, 1.187, 4.27994, 3.19834, -4.16857, -2.42796, -3.93363, -1.76999, -4.63881, -3.62552, 0.000772253]
    #true_UPp 1.00001035485
    #UIVarVals = [0.373115205354,-0.379193401074,1.00173425967,-0.00264995824584,0.175911687296,0.923233258521,-0.0025446101209,-0.386704788202,-0.066415425898,-0.386801405334,0.0718572381342,0.00414882783499,-0.088760753235,0.138624959898,0.0419080851022,0.141712125722,-0.0485849164221,0.0497897294305,-0.0389565896148,0.00190400835361,-0.376173454762,0.0692168556626,-0.0690749599181,0.0101365562385,-0.0523857160345,0.0380928513001]
    #scipyInvCovMatrix = scipyTotNomInvCovMatrix
elif FIT_TO_DIRECT_MATRIX:
    scipyInvCovMatrix = scipyDirectInvCovMatrix
    '''
    print " ---------- Matt's debug ---------- Bellis "
    for i in range(0,26):
        for j in range(0,26):
            num = -999.99999999999999
            if UsIsCorMatrix[i][j]!=0.0:
                num = nominalUsIsCorrMatrixStat[i][j]/UsIsCorMatrix[i][j]
            #othernum = scipyTotNomInvCovMatrix[i][j]/scipyDirectInvCovMatrix[i][j]
            othernum = -999.999999999
            if UsIsCovMatrix[i][j]!=0.0:
                othernum = nominalUsIsCovMatrixStat[i][j]/UsIsCovMatrix[i][j]
            #print "%10.5f %10.5f %10.5f\t%10.5f %10.5f %10.5f" % (scipyTotNomInvCovMatrix[i][j],scipyDirectInvCovMatrix[i][j],othernum,nominalUsIsCorrMatrixStat[i][j],UsIsCorMatrix[i][j], num)
            print "%10.5f %10.5f %10.5f\t%10.5f %10.5f %10.5f" % (nominalUsIsCovMatrixStat[i][j],UsIsDirectCovMatrix[i][j],othernum,nominalUsIsCorrMatrixStat[i][j],UsIsCorMatrix[i][j], num)

            ########## Testing out if this makes a difference ##########
            # Fails #
            if UsIsCovMatrix[i][j]!=0.0:
                scipyInvCovMatrix[i][j] /= othernum
    '''


print " -------- Matt's debug scipy -----------"
print scipyInvCovMatrix
print " -------- Matt's debug scipy -----------"
print " -------- Matt's debug uivarvals -----------"
print UIVarVals 
print " -------- Matt's debug uivarvals -----------"


################################################################################
# Returns rho amplitudes.
################################################################################
def ampsFromTreesPenguins(alpha,beta,Tp,Tm,Tz,Pp,Pm):
    Pz = -0.5*(Pp+Pm)
    
    eToNegIAlpha = cos(-1*alpha) + 1j * sin(-1*alpha)
    eToPosIAlpha = cos(alpha) + 1j * sin(alpha)
    eToTwoIBeta = cos(2*beta) + 1j * sin(2*beta)
    
    Aplus = Tp * eToNegIAlpha + Pp
    Aminus = Tm * eToNegIAlpha + Pm
    Azero = Tz * eToNegIAlpha + Pz
    
    Abplus = Tm * eToPosIAlpha + Pm*eToTwoIBeta
    Abminus = Tp * eToPosIAlpha + Pp*eToTwoIBeta
    Abzero = Tz * eToPosIAlpha + Pz*eToTwoIBeta

#    print [abs(Aplus),abs(Aminus),abs(Azero),abs(Abplus),abs(Abminus),abs(Abzero)]

    return [Aplus,Aminus,Azero,Abplus,Abminus,Abzero]

################################################################################
# Returns Us and Is from rho amplitudes.
# Uses output from previous function. (ampsFromTreesPenguins)
################################################################################
def UsIsFromAmps(Ap,Am,Az,Abp,Abm,Abz):

    U_p_p = abs(Ap)**2 + abs(Abp)**2
    U_p_m = abs(Am)**2 + abs(Abm)**2
    U_p_z = abs(Az)**2 + abs(Abz)**2
    
    U_m_p = abs(Ap)**2 - abs(Abp)**2
    U_m_m = abs(Am)**2 - abs(Abm)**2
    U_m_z = abs(Az)**2 - abs(Abz)**2
    
    U_pRe_pm = (Ap*Am.conjugate() + Abp*Abm.conjugate()).real
    U_pRe_pz = (Ap*Az.conjugate() + Abp*Abz.conjugate()).real
    U_pRe_mz = (Am*Az.conjugate() + Abm*Abz.conjugate()).real
    
    U_mRe_pm = (Ap*Am.conjugate() - Abp*Abm.conjugate()).real
    U_mRe_pz = (Ap*Az.conjugate() - Abp*Abz.conjugate()).real
    U_mRe_mz = (Am*Az.conjugate() - Abm*Abz.conjugate()).real
    
    U_pIm_pm = (Ap*Am.conjugate() + Abp*Abm.conjugate()).imag
    U_pIm_pz = (Ap*Az.conjugate() + Abp*Abz.conjugate()).imag
    U_pIm_mz = (Am*Az.conjugate() + Abm*Abz.conjugate()).imag
    
    U_mIm_pm = (Ap*Am.conjugate() - Abp*Abm.conjugate()).imag
    U_mIm_pz = (Ap*Az.conjugate() - Abp*Abz.conjugate()).imag
    U_mIm_mz = (Am*Az.conjugate() - Abm*Abz.conjugate()).imag
    
    Ip = (Abp*Ap.conjugate()).imag
    Im = (Abm*Am.conjugate()).imag
    Iz = (Abz*Az.conjugate()).imag
    
    I_Re_pm = (Abp*Am.conjugate() - Abm*Ap.conjugate()).real
    I_Re_pz = (Abp*Az.conjugate() - Abz*Ap.conjugate()).real
    I_Re_mz = (Abm*Az.conjugate() - Abz*Am.conjugate()).real
    
    I_Im_pm = (Abp*Am.conjugate() + Abm*Ap.conjugate()).imag
    I_Im_pz = (Abp*Az.conjugate() + Abz*Ap.conjugate()).imag
    I_Im_mz = (Abm*Az.conjugate() + Abz*Am.conjugate()).imag

    curUsIsDictionary = {}
    curUsIsDictionary["UPp"]     = U_p_p
    curUsIsDictionary["UPm"]     = U_m_p
    curUsIsDictionary["UMm"]     = U_m_m
    curUsIsDictionary["UMp"]     = U_p_m
    curUsIsDictionary["U0m"]     = U_m_z
    curUsIsDictionary["U0p"]     = U_p_z
    curUsIsDictionary["UPMpRe"]  = U_pRe_pm
    curUsIsDictionary["UPMmRe"]  = U_mRe_pm
    curUsIsDictionary["UP0pRe"]  = U_pRe_pz
    curUsIsDictionary["UP0mRe"]  = U_mRe_pz
    curUsIsDictionary["UM0pRe"]  = U_pRe_mz
    curUsIsDictionary["UM0mRe"]  = U_mRe_mz
    curUsIsDictionary["UPMpIm"]  = U_pIm_pm
    curUsIsDictionary["UPMmIm"]  = U_mIm_pm
    curUsIsDictionary["UP0pIm"]  = U_pIm_pz
    curUsIsDictionary["UP0mIm"]  = U_mIm_pz
    curUsIsDictionary["UM0pIm"]  = U_pIm_mz
    curUsIsDictionary["UM0mIm"]  = U_mIm_mz
    curUsIsDictionary["IP"]      = Ip
    curUsIsDictionary["IM"]      = Im
    curUsIsDictionary["I0"]      = Iz
    curUsIsDictionary["IPMRe"]   = I_Re_pm
    curUsIsDictionary["IP0Re"]   = I_Re_pz
    curUsIsDictionary["IM0Re"]   = I_Re_mz
    curUsIsDictionary["IPMIm"]   = I_Im_pm
    curUsIsDictionary["IP0Im"]   = I_Im_pz
    curUsIsDictionary["IM0Im"]   = I_Im_mz

#    for curUIName in UIVars:
#        print curUIName,"\t",curUsIsDictionary[curUIName]

#    print U_p_p
#    print U_m_p

    return curUsIsDictionary








################################################################################
# 
################################################################################
def calcChi2(cur_alpha,cur_beta,cur_Tp,cur_Tm,cur_Tz,cur_Pp,cur_Pm):
    ampList = ampsFromTreesPenguins(cur_alpha,cur_beta,cur_Tp,cur_Tm,cur_Tz,cur_Pp,cur_Pm)
    curUsIsDict = UsIsFromAmps(ampList[0],ampList[1],ampList[2],ampList[3],ampList[4],ampList[5])

    curUsIsList = []
    curDiffList = []
    for i in range(0,numUsIs):
        curName = UIVars[i]
        curVal = curUsIsDict[curName]
        fitVal = UIVarVals[i]
        curUsIsList.append(curVal)
        curDiffList.append(fitVal-curVal)

    curUPpVal = curUsIsDict["UPp"]

    tempVec = numpy.array(curDiffList)
    diffVec = mat(tempVec)

    chi2 = diffVec * scipyInvCovMatrix * diffVec.T
    #print chi2
    chi2 = chi2[0,0]

    return chi2, curUPpVal


################################################################################
# For Minuit 
################################################################################
def fcn( npar, gin, f, par, iflag ):
     global ncount

     # calculate chisquare
     Tp_scan = par[0] + 1j*par[1]  
     Tm_scan = par[2] + 1j*par[3]
     Tz_scan = par[4] + 1j*par[5]
     Pp_scan = par[6] + 1j*par[7]
     Pm_scan = par[8] + 1j*par[9]
     beta_scan = par[10]

#     print "par[0] =", par[0] 
#     print "par[1] =", par[1] 
#     print "par[2] =", par[2] 
#     print "par[3] =", par[3] 
#     print "par[4] =", par[4] 
#     print "par[5] =", par[5] 
#     print "par[6] =", par[6] 
#     print "par[7] =", par[7] 
#     print "par[8] =", par[8] 
#     print "par[9] =", par[9] 
#     print "par[10]=", par[10]

     chisq,UPp_scan = calcChi2(alphaScanVal,
                               beta_scan,
                               Tp_scan,
                               Tm_scan,
                               Tz_scan,
                               Pp_scan,
                               Pm_scan)

#     quit()

     # Need gaussian constraint on UPp_scan so it is close to 1.0

     chisq = float(chisq)
     if CONSTRAIN_UPP:
         epsilon = G_EPSILON
         constraint = ((UPp_scan-1.0)/epsilon)**2
         #print "MYCHI2: %f %f %f" % (chisq, constraint,chisq+constraint)
         chisq += constraint

#     print UPp_scan
#     print chisq

     f[0] = chisq

     ncount += 1



# void mnparm(Int_t k, TString cnamj, Double_t uk, Double_t wk, Double_t a, Double_t b, Int_t& ierflg)
# 
# Implements one parameter definition*-*-*-
# *-*              ===================================
# *-*        Called from MNPARS and user-callable
# *-*    Implements one parameter definition, that is:
# *-*          K     (external) parameter number
# *-*          CNAMK parameter name
# *-*          UK    starting value
# *-*          WK    starting step size or uncertainty
# *-*          A, B  lower and upper physical parameter limits
# *-*    and sets up (updates) the parameter lists.
# *-*    Output: IERFLG=0 if no problems
# *-*                  >0 if MNPARM unable to implement definition
# *



########################################################################################
# Fit
########################################################################################

def testfit(initVector):

    gMinuit = TMinuit(11)
    gMinuit.SetFCN( fcn )
 
    ierflg = Long(1982) # ROOT.Long
 
    # SET ERRordef <up>
    # SET ERRordef: Sets the value of UP (default value= 1.), defining parameter
    # errors. Minuit defines parameter errors as the change in parameter value
    # required to change the function value by UP. Normally, for chisquared fits
    # UP=1, and for negative log likelihood, UP=0.5. 
    UPlist = array( 'd', 1*[1.] )
    if isChiSq:
       UPlist[0] = 1
    else:
       UPlist[0] = 0.5
    gMinuit.mnexcm( "SET ERR", UPlist, 1, ierflg )
 
    # Set starting values and step sizes for parameters

    initValues = initVector
#    for i in range(0,10):
##        initValues.append(random.uniform(-1.0,1.0))
#        initValues.append(random.uniform(-20.0,20.0))
##    initValues.append(21*pi/180.0)
#    initValues.append(random.uniform(0,pi))
##    initValues.append(3.04)
    initValues = tuple(initValues)

    vstart = array( 'd', initValues )

    step   = array( 'd', ( 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1 ) )
 
    gMinuit.mnparm( 0, "Tp_re", vstart[0], step[0], 0, 0, ierflg )
    gMinuit.mnparm( 1, "Tp_im", vstart[1], step[1], 0, 0, ierflg )

    gMinuit.mnparm( 2, "Tm_re", vstart[2], step[2], 0, 0, ierflg )
    gMinuit.mnparm( 3, "Tm_im", vstart[3], step[3], 0, 0, ierflg )

    gMinuit.mnparm( 4, "Tz_re", vstart[4], step[4], 0, 0, ierflg )
    gMinuit.mnparm( 5, "Tz_im", vstart[5], step[5], 0, 0, ierflg )

    gMinuit.mnparm( 6, "Pp_re", vstart[6], step[6], 0, 0, ierflg )
    gMinuit.mnparm( 7, "Pp_im", vstart[7], step[7], 0, 0, ierflg )

    gMinuit.mnparm( 8, "Pm_re", vstart[8], step[8], 0, 0, ierflg )
    gMinuit.mnparm( 9, "Pm_im", vstart[9], step[9], 0, 0, ierflg )
 
#    gMinuit.mnparm( 10, "beta", vstart[10], step[10], 0, pi, ierflg )
    gMinuit.mnparm( 10, "beta", vstart[10], step[10], 0, 2*pi, ierflg )
 
    # Now ready for minimization step
 
    # Perform minuit fit
    arglist = array( 'd', 2*[0.] )
#    arglist[0] = 1500 # Maximum number of steps
    arglist[0] = CALLLIMIT # Maximum number of steps
    arglist[1] = TOLERANCE # Tolerance
    gMinuit.mnexcm( "MIGRAD", arglist, 2, ierflg )

########
########
########
#    gMinuit.mnexcm( "IMPROVE", arglist, 2, ierflg )
########
########
########

#    print "MIGRAD_ERROR_FLAG:", ierflg

    migradErrorFlag = int(ierflg * 1)

    # Calculate symmetric errors
    hesseSteps = array( 'd', 1*[0.] )
    hesseSteps[0] = 1000 # Maximum number of steps
    gMinuit.mnexcm( "HESSE", hesseSteps, 2, ierflg )

#    print "HESSE_ERROR_FLAG:", ierflg

    hesseErrorFlag = int(ierflg * 1)

    ## Calculate asymmetric errors
    #   minosSteps = array( 'd', 1*[0.] )
    #   minosSteps[0] = 1000 # Maximum number of steps
    #   gMinuit.mnexcm( "MINOS", minosSteps, 2, ierflg )
 
    # Print results
    amin, edm, errdef = Double(0.18), Double(0.19), Double(0.20)
    nvpar, nparx, icstat = Long(1983), Long(1984), Long(1985)
    gMinuit.mnstat( amin, edm, errdef, nvpar, nparx, icstat )
    gMinuit.mnprin( 3, amin )
 
    finalPars = []
    finalParsErr = []
    dumx, dumxerr = Double(0), Double(0)
    for i in range(0,11):
       gMinuit.GetParameter(i,dumx, dumxerr)
       finalPars.append(float(dumx))
       finalParsErr.append(float(dumxerr))


    Tp_temp = finalPars[0] + 1j*finalPars[1]  
    Tm_temp = finalPars[2] + 1j*finalPars[3]
    Tz_temp = finalPars[4] + 1j*finalPars[5]
    Pp_temp = finalPars[6] + 1j*finalPars[7]
    Pm_temp = finalPars[8] + 1j*finalPars[9]
    beta_temp = finalPars[10]
    tempchisq,UPp_temp = calcChi2(alphaScanVal,
                                  beta_temp,
                                  Tp_temp,
                                  Tm_temp,
                                  Tz_temp,
                                  Pp_temp,
                                  Pm_temp)
    #tempchisq = float(tempchisq)
    tempchisq = float(tempchisq)
    if CONSTRAIN_UPP:
        tempepsilon = G_EPSILON
        tempchisq = tempchisq + ((UPp_temp-1.0)/tempepsilon)**2

    print "Final chisq =", tempchisq
    print "UPp =", UPp_temp
    print "GREPME\t"+str(alphaInDeg)+"\t"+str(tempchisq)

    return [tempchisq,
            migradErrorFlag,
            hesseErrorFlag,
            [finalPars[0],
             finalPars[1],
             finalPars[2],
             finalPars[3],
             finalPars[4],
             finalPars[5],
             finalPars[6],
             finalPars[7],
             finalPars[8],
             finalPars[9],
             finalPars[10]]]
 
#    print "a1:\t", finalPars[0], "+/-", finalParsErr[0]
#    print "a2:\t", finalPars[1], "+/-", finalParsErr[1]
#    print "a3:\t", finalPars[2], "+/-", finalParsErr[2]
#    print "a4:\t", finalPars[3], "+/-", finalParsErr[3]

########################################################################################







ncount = 0
bestStartingVector = [ 3.34483e+00,
                       1.18700e+00,
                       4.27994e+00,
                       3.19834e+00,
                       -4.16857e+00,
                       -2.42796e+00,
                       -3.93363e+00,
                       -1.76999e+00,
                       -4.63881e+00,
                       -3.62552e+00,
                       7.72253e-04]

##bestStartingVector = [9.4680641787, -4.8231015457, 15.53605429, -15.834539936, 13.348747338, 13.698083258, -15.359843768, 16.614598598, 11.360984585, 16.584915767, 0.88444855997]
#bestStartingVector = [9.46806417873, -4.82310154575, 15.536054292, -15.8345399363, 13.3487473382, 13.6980832586, -15.3598437686, 16.6145985981, 11.3609845853, 16.5849157674, 0.884448559971]
#
#
#bestStartingVector = [0.86919920805416572840, 0.88984699520751087132 ,0.67414842631251459260, 0.51751372931539973976, 0.50709785232256909815, 0.75330479485166645937, 0.77400399032580602388, 0.56250101130235541369, 0.81547322826661439166, 0.52605542442716712870, 1.50193050800646776821]
#

# Nominal 89 degree best fit starting vector:
nominalEightyNineDegreeFitVec = [0.47340320893662735102,-0.24115507728742979765,0.77680271460228844660,-0.79172699681323921439,0.66743736690863642025,0.68490416293001943338,-0.76799218843247363253,0.83072992990630112331,0.56804922926431133945,0.82924578837081552862,0.88444855997106464063]
bestStartingVector = [0.47340320893662735102,-0.24115507728742979765,0.77680271460228844660,-0.79172699681323921439,0.66743736690863642025,0.68490416293001943338,-0.76799218843247363253,0.83072992990630112331,0.56804922926431133945,0.82924578837081552862,0.88444855997106464063]
# Nominal 89 deg best fit errors
bestErrorVector = [9.24682e-04,7.33422e-04,1.62094e-03,1.24732e-03,7.58577e-02,7.48185e-02,7.23654e-04,9.45398e-04,1.50376e-03,1.31166e-03,6.27915e-04]
# GREPME899.24243246592
# GREPVEC -0.25927748481547036041,-1.17237445360151637530,0.78922023711394873047,-0.26502169406873693491,-0.22547911773770989985,0.23740529596914700106,0.49013755167604639018,0.28662700146034258974,-0.44117402199969307786,0.69164836449372646410,0.80734295354655294386,
# GREPRESULTS89 9.24243246592 0 0 NUMREP 1

startingVector = bestStartingVector

previousChi2 = 0


# Nominal 89 degree best fit vector:
# GREPSTARTINGVEC 9.46806417873 -4.82310154575 15.536054292 -15.8345399363 13.3487473382 13.6980832586 -15.3598437686 16.6145985981 11.3609845853 16.5849157674 0.884448559971
# GREPVEC -0.939166884529 -13.1352083225 18.1225779196 -0.726322746072 -8.47551741297 6.46202195781 12.3873480544 -0.301835547156 -0.257156467682 17.9207035997 1.52226407433
# Chi2 = 9.24244021286
# [-0.939166884529, -13.1352083225, 18.1225779196, -0.726322746072, -8.47551741297, 6.46202195781, 12.3873480544, -0.301835547156, -0.257156467682, 17.9207035997, 1.52226407433]
#
# Result of fitting new run1-5 log at 89 degrees starting at the above vector values:
# GREPVEC -3.33884877856 -12.8529927261 19.5693975823 0.701550491018 -8.09078350234 5.71721153833 12.2607267737 -2.58441317135 -1.66897708821 19.3617861819 1.52350691805 
# GREPRESULTS89 60.4204950681 0 0 NUMREP 1


################################################################################
# Here's the loop. 
################################################################################
for i in [89]:
#for i in range(89,0,-1):
#for i in [89]:
#for i in range(0,180+1):
#for i in [0]:
#for i in range(0,5):

    alphaInDeg = i
    alphaScanVal = alphaInDeg*pi/180.0
    migradFlag = -1

    continueScan = False
    nRepeat = NUMREPITITIONS
    curMinBadChi2 = 99999999999
    curMinChi2    = 99999999999
    curNumRepititions = 0
    while not continueScan:

        ncount = 0


        print "GREPSTARTINGVEC\t",
        for j in range(0,len(startingVector)):
#            print startingVector[j],"\t",
            print "%20.20f,\t" % (startingVector[j]),
        print ""
        
        curChiSq, migradFlag, hesseFlag, curFitVector = testfit(startingVector)

        if migradFlag == 0:
            curNumRepititions += 1
            
            if curChiSq < curMinChi2:
                curMinChi2 = curChiSq
                bestStartingVector = curFitVector
        else:
            if curChiSq < curMinBadChi2:
                curMinBadChi2 = curChiSq

        if curNumRepititions>=nRepeat and curMinChi2<=curMinBadChi2:
            continueScan = True
            startingVector = bestStartingVector

#        curDiff = abs(curChiSq-previousChi2)
#        if i>0 and curDiff>3:
#            continueScan = False

        if migradFlag == 0:

            print "GREPVEC\t",
            for j in range(0,len(curFitVector)):
                print "%20.20f,\t" % (curFitVector[j]),
                #print curFitVector[j],"\t",
            print ""

            print "GREPRESULTS\t",alphaInDeg,"\t",curChiSq,"\t",migradFlag,"\t",hesseFlag,"\tNUMREP",curNumRepititions
        else:
            print "GREPRESULTS\t",alphaInDeg,"\t",curChiSq,"\t",migradFlag,"\t",hesseFlag,"\tBADFIT"

        if continueScan:
            print "GREPRESULTSBEST\t",alphaInDeg,"\t",curMinChi2
            previousChi2 = curMinChi2

        if not continueScan:
            randVec = []
            for j in range(0,10):
                ################################################################
                # Setting the range for the amplitudes.
                ################################################################
                randVec.append(random.uniform(-1.0,1.0))
#            for j in range(0,11):
#                centralVal = nominalEightyNineDegreeFitVec[j]
#                randVal = random.gauss(centralVal,bestErrorVector[j]*2)
#                randVec.append(randVal)

#            randVec.append(random.uniform(0,pi))
            randVec.append(random.uniform(0,2*pi))
            startingVector = randVec
