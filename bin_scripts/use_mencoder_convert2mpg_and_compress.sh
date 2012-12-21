#!/bin/tcsh

set input_file = $1

mencoder $input_file -of mpeg -mpegopts format=mpeg1:tsaf:muxrate=2000 -o output.mpg -oac lavc -lavcopts acodec=mp2:abitrate=224 -ovc lavc -lavcopts vcodec=mpeg1video -vf scale=800:600
