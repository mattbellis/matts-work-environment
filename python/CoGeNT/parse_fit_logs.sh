#!/usr/bin/env tcsh 

foreach file( $* )
  #echo $file
  set xsec = `grep "sigma_n = " $file | awk '{print $6}'`
  set mDM = `grep 'mDM = ' $file | awk '{print $6}'`
  set lh = `grep 'fval' $file | tail -1 | awk '{print $3}'`
  echo $xsec" "$mDM" "$lh
end
