#!/bin/tcsh

set file = $1

latex $file
dvips -t letter $file -o
