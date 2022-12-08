# Day 7

## Part A
python day7.py INPUT_day7.txt | grep close | grep THIS | awk '{print $7}' | paste -sd+ - | bc

## Part B
# Part B
rootsize =  50216456
totaldisk = 70000000

need_to_free = 30000000 - (totaldisk - rootsize)
print(f"Need to free: {need_to_free}")
# 10216456

python day7.py INPUT_day7.txt | grep close | grep -v THIS | sort -k5 -n


