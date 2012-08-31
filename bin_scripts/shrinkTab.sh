#!/bin/tcsh -f

a2ps --columns=2 -L100 -o `basename $1 $2`ps $1
