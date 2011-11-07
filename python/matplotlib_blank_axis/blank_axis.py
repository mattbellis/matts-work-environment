#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

################################################################################
# Plot
################################################################################
def main():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0,20)
    ax.set_ylim(-1,1)
    ax.grid()
    ax.set_xlabel('Position in room',size=20)
    ax.set_ylabel('Probability (arbitrary units)',size=20)

    fig.savefig("test.png")

    plt.show()

################################################################################

if __name__ == '__main__':
    main()

