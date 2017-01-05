#!/bin/tcsh

set name = `echo $1 | awk -F'.' '{print $1}'`
set ext  = `echo $1 | awk -F'.' '{print $2}'`
echo convert -flatten $1 $name"_flat."$ext
     convert -flatten $1 $name"_flat."$ext
