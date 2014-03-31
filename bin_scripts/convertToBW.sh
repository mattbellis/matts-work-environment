#!/bin/tcsh -f

foreach file($*)
  set newfile = `basename $file .jpg`"_BW.jpg"
  echo $file $newfile
  convert $file temp0.ppm
  ppmquant -fs 255 temp0.ppm > temp1.ppm
  ppmdist temp1.ppm > temp2.ppm
  convert temp2.ppm $newfile
  rm temp0.ppm temp1.ppm temp2.ppm >& /dev/null
end
