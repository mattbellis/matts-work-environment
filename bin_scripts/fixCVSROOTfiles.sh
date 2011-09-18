#!/bin/tcsh

cd $CLAS_PACK

foreach file(`find * | grep 'CVS/Root'`)
  echo $file
  echo bellis@cvs.jlab.org:/group/clas/clas_cvs > $file
  more $file
end

