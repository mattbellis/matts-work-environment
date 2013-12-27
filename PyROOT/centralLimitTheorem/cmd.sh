#./plotOneExample.py 1   1000   100 100 test True
#./plotOneExample.py 100 10000 1000 100 test True



# Flat distribution

set tag = "clt"
#foreach set( "0" "0,2,4" "0,2" "1" "2" "3" "4" "5" "1,4" "1,2,3,5" )
foreach set( "6,7,8" )
################### nsamp  ntrials  nstep  nbins  whichfuncs   tag
./plotOneExample.py 1        100      1     100   $set         $tag                      batch
./plotOneExample.py 1        1000     10    100   $set         $tag                      batch
./plotOneExample.py 1        10000    100   100   $set         $tag                      batch
./plotOneExample.py 2        100      1     100   $set         $tag                      batch
./plotOneExample.py 2        10000    100   100   $set         $tag                      batch
./plotOneExample.py 5        100      1     100   $set         $tag                      batch
./plotOneExample.py 5        10000    100   100   $set         $tag                      batch
./plotOneExample.py 10       100      1     100   $set         $tag                      batch
./plotOneExample.py 10       10000    100   100   $set         $tag                      batch
./plotOneExample.py 100      10000    100   100   $set         $tag                      batch

end

#./plotOneExample.py 1        100      1     100   "0,2,4"  test                      batch

