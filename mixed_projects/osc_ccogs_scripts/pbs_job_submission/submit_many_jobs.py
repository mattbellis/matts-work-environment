import sys
import subprocess as sp

for i in range(0,10):
    for j in range(i,10):
        
        #file0 = "flat_10M_arcseconds_max1000000_index%03d.dat" % (i)
        #file1 = "flat_10M_arcseconds_max1000000_index%03d.dat" % (j)
        file0 = "flat_100k_arcseconds_max10000_index%03d.dat" % (i)
        file1 = "flat_100k_arcseconds_max10000_index%03d.dat" % (j)

        tag = "d100k_f1M_%03d_%03d" % (i,j)

        cmd = ['python','build_submission_files.py',tag, file0,file1,'flatflat']

        print cmd
        sp.Popen(cmd,0).wait()


