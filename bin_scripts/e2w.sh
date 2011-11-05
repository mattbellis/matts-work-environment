#!/bin/tcsh

echo $1 | awk '{print sqrt(($1 + 0.938)*($1 + 0.938) - ($1)*($1))}'
#echo $1 | awk '{print sqrt(($1 + 2*0.938)*($1 + 2*0.938) - ($1)*($1))}'


