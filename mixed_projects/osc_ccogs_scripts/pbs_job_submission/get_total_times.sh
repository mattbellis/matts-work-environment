grep '0pf+0w' $* | awk '{print $1}' | awk -F":" '{print $3}' | awk -Fu '{print $1}' | sort -n > total_times.txt
awk '{x += $1} END {print "Sum: "x}' total_times.txt
