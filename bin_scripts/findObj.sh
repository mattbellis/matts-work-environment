#!/bin/csh -f

set pattern=$1;
shift;

foreach file ($*)
	echo 'Searching '$file '...'
#	nm $file | c++filt | grep $pattern | grep '^[0-9a-z	]*T'
	nm $file | c++filt | grep $pattern 
end

