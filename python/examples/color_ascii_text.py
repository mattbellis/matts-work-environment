#!/usr/bin/env python 

import sys

################################################################################
# You can find a nice reference for the colors here, as well as other sites
# online. 
#
# http://bashscript.blogspot.com/2010/01/shell-colors-colorizing-shell-scripts.html
#
################################################################################
def main():

    output = ""
    for i in range(0,15):
        for j in range(0,15):
            fgcolor = 31 + i%8 
            bgcolor = 41 + j%8

            output += "\033[%dm\033[%dm%3d \033[0m" % (bgcolor,fgcolor,i+j)

        output += "\n"

    print output



################################################################################
################################################################################
if __name__ == '__main__':
    main()


