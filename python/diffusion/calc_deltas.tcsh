sdiff $1 $2 | awk '{print $1 " " ($5/$2-1)*1000}'
