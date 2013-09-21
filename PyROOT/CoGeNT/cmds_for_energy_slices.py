#!/usr/bin/env python

import subprocess as sp
import sys

################################################################################
# Do the bulk of all the fits.
################################################################################

mod_options = [[''],['--exp-mod'],['--flat-mod'],['--cg-mod'],['--flat-mod --exp-mod'],['--cg-mod --flat-mod'],['--cg-mod --flat-mod --exp-mod']]
mod_flag = [0,1,2,3,4,5,6]


lo_ecut = 1.5

hi_ecut = 0.8

cut_width = 0.4
if len(sys.argv)>1:
    cut_width = float(sys.argv[1])

while ( lo_ecut<2.8 and hi_ecut<3.2 ):

    hi_ecut = lo_ecut + cut_width

    lo_ecut_s = "%2.1f" % (lo_ecut)
    hi_ecut_s = "%2.1f" % (hi_ecut)

    for i,mod in enumerate(mod_options):

        count = 1
      
        log_file_name = "log_energy_slices_lo%s_hi%s_mod%d.log" % (lo_ecut,hi_ecut,i)

        cmd = ['python2.7','read_in_from_text_file.py']
        cmd += ['data/before_fire_LG.dat']
        cmd += ['-b']

        if lo_ecut<1.5:
            cmd += ['--gc-flag','2','--add-gc']

        run_fit = True
        if lo_ecut<1.5 and i!=0:
            cmd += mod
        elif lo_ecut<1.5 and i==0:
            cmd 
        elif lo_ecut>=1.5 and (i==0 or i==2):
            cmd += ['--no-exp','--no-cg']
            if i==2:
                cmd += mod
        else:
            run_fit = False

        cmd += ['--e-lo',lo_ecut_s]
        cmd += ['--e-hi',hi_ecut_s]

        if run_fit:
            outfile = open(log_file_name,"w")
            #print cmd
            output = ""
            for c in cmd:
                output += "%s " % (c)
            output += "\n\n\n"
            outfile.write(output)
            print output

            output = sp.Popen(cmd, stdout=sp.PIPE).communicate()[0]
            outfile.write(output)
            outfile.close()

            #print output

            #exit(0)

    lo_ecut += 0.1
