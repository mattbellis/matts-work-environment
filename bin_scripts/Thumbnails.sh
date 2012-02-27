#!/bin/tcsh


set size = $1

shift

set format = $1

shift

foreach file($*) 

  set newfile = "small_"`basename $file jpg`""$format
  #set newfile = `basename $file .jpg`"small."$format

  echo convert -size $size"x"$size $file -resize $size"x"$size +profile '"*"' $newfile
       convert -size $size"x"$size $file -resize $size"x"$size +profile "*" $newfile

end
