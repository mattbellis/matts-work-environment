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
    #ax.set_xlim(-3,3)

    ax.set_ylim(-1,1)
    ax.grid()

    ax.set_xlabel('Position in room (meters)',size=20)
    #ax.set_xlabel('x-component of momentum (m/s)',size=20)
    ax.set_ylabel('Probability (arbitrary units)',size=20)

    fig.savefig("prob_1.png")

    plt.show()

################################################################################

if __name__ == '__main__':
    main()

