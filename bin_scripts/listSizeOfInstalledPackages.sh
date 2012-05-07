#!/bin/tcsh -f

foreach package ( `dpkg-query -W | awk '{print $1}'` )
  set size = `dpkg-query -p $package | grep 'Installed-Size' | awk '{print $2}'`
  echo $package "\t" $size
end

