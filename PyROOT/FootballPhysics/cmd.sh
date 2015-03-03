#!/bin/tcsh 

#./look_at_data.py all_scores.txt --batch --max 0    --tag football_max0
#./look_at_data.py all_scores.txt --batch --max 10    --tag football_max10
#./look_at_data.py all_scores.txt --batch --max 100    --tag football_max100
#./look_at_data.py all_scores.txt --batch --max 300    --tag football_max300
#./look_at_data.py all_scores.txt --batch --max 200    --tag football_max200
#./look_at_data.py all_scores.txt --batch --max 500    --tag football_max500
#./look_at_data.py all_scores.txt --batch --max 1000    --tag football_max1000
#./look_at_data.py all_scores.txt --batch --max 10000    --tag football_max10000

./look_at_data.py all_scores.txt --batch --max 0 --tag football_nbins50_max0
./look_at_data.py all_scores.txt --batch --max 10000 --tag football_nbins50_max10000
./look_at_data.py all_scores.txt --batch --max 10000 --nbins 40 --tag football_nbins40_max10000
./look_at_data.py all_scores.txt --batch --max 10000 --nbins 30 --tag football_nbins30_max10000
./look_at_data.py all_scores.txt --batch --max 10000 --nbins 20 --tag football_nbins20_max10000
./look_at_data.py all_scores.txt --batch --max 10000 --nbins 10 --tag football_nbins10_max10000

