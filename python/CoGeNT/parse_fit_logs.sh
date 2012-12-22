#!/usr/bin/env tcsh 

foreach file( $* )
  #echo $file
  set xsec = `grep sigma_n $file | awk '{print $3}'`
  set mDM = `grep '23  mDM' $file | awk '{print $3}'`
  set lh = `grep 'vals' $file | tail -1 | awk '{print $5}'`
  echo $xsec" "$mDM" "$lh
end
