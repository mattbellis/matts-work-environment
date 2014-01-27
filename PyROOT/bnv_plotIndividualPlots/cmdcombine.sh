#!/usr/bin/tcsh 

set type = "shape"

echo $type

./combineFilesOfHistos.py rootFiles/h"$type"_AllSP_shapeStudyCuts1.root \
                          rootFiles/h"$type"_SP1005_shapeStudyCuts1.root 0.50 \
                          rootFiles/h"$type"_SP998_shapeStudyCuts1.root 0.50 \
                          rootFiles/h"$type"_SP1237_shapeStudyCuts1.root 0.35 \
                          rootFiles/h"$type"_SP1235_shapeStudyCuts1.root 0.35 
