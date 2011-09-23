#!/bin/tcsh 

set isBeamer = 0

if ( $1 == "beamer" ) then
  set isBeamer = 1
  shift
endif

set sizewhich = $1
set sizenum = $2

shift
shift

#echo "isBeamer: " $isBeamer

@ count = 0

foreach file($*)
  echo
  if ( $isBeamer == 1 ) then
    echo "\\frame[T]"
    echo "{"
    echo "\\frametitle{XXX}"
  endif

  echo "\\begin{figure}[H]"
  echo "\\includegraphics[$sizewhich=$sizenum\\text$sizewhich]{$file}"
  echo "\\label{lab$count}"
  echo "\\caption{Caption goes here}"
  echo "\\end{figure}"
  if ( $isBeamer == 1 ) then
    echo "}"
  endif
  echo
  @ count += 1
end
