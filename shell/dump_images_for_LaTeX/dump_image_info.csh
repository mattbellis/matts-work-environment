#!/bin/tcsh 

set isBeamer = 0

if ( $1 == "beamer" ) then
  set isBeamer = 1
  shift
endif

#set sizewhich = $1
#set sizenum = $2

set sizewhich = 'width'

#shift
#shift

#echo "isBeamer: " $isBeamer

@ count = 0

foreach file($*)

    @ width = `extractbb -O $file | grep BoundingBox | grep -v HiRes | awk '{print $4}'`
    @ height = `extractbb -O $file | grep BoundingBox | grep -v HiRes | awk '{print $5}'`

    set ratio = `echo $width $height | awk '{print $1/$2}'`
    set sizenum = `echo $ratio | awk '{if($1>1.5){print 0.9}else{print 0.9*$1/1.5}}'`

    #echo $width" "$height" "$ratio

  echo
  if ( $isBeamer == 1 ) then
    echo "\frame"
    echo "{"
    echo "\frametitle{Thermal Physics - Chapter 1}"
  endif

  echo "\begin{figure}[H]"
  #echo "\fcolorbox{white}{white}{"
  echo "\includegraphics[$sizewhich=$sizenum\text$sizewhich]{$file}"
  #echo "}"
  echo "\label{lab$count}"
  #echo "\caption{Caption goes here}"
  echo "\end{figure}"
  if ( $isBeamer == 1 ) then
    echo "}"
  endif
  echo
  @ count += 1
end
