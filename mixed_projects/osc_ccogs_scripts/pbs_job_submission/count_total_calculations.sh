grep  'total:' $* | awk '{x += $2} END {print "Sum: "x}'
