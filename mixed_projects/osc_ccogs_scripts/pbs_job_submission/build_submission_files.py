import sys
import subprocess as sp

tag = sys.argv[1]
data = sys.argv[2]
flat = sys.argv[3]
flatflag = None
if len(sys.argv)>4:
    flatflag = sys.argv[4]

pbs_file_name = "pbs_%s.pbs" % (tag)
pbs_file = open(pbs_file_name,'w+')

################################################################################
# Boiler plate
################################################################################
output = ""
output += "#PBS -N ccogs:%s\n" % (tag)
output += '#PBS -l walltime=48:00:00\n'
#output += '#PBS -l nodes=1:ppn=12:gpus=2\n'
output += '#PBS -l nodes=1:ppn=1:gpus=2\n'
output += '#PBS -j oe\n\n'

output += '#PBS -S /bin/bash\n\n'

output += 'module load cuda\n'
output += 'module list\n\n'

output += 'cd $PBS_O_DIR # Note that this is not WORKDIR\n'
output += 'ls -ltr\n'

output += "pbsdcp ~ucn1219/ccogs/angular_correlation/bin/angular_correlation $TMPDIR\n"
######output += "pbsdcp ~ucn1219/pbs_job_submission/run_GPU_calculation_2devices.csh $TMPDIR\n"
output += "pbsdcp ~ucn1219/pbs_job_submission/run_GPU_calculation.csh $TMPDIR\n"
#output += "pbsdcp ~ucn1219/data/num_flat_test/%s $TMPDIR\n" % (data)
#output += "pbsdcp ~ucn1219/data/num_flat_test/%s $TMPDIR\n\n" % (flat)
output += "pbsdcp ~ucn1219/data/z_slices/%s $TMPDIR\n" % (data)
output += "pbsdcp ~ucn1219/data/z_slices/%s $TMPDIR\n\n" % (flat)

output += "cd $TMPDIR\n"
output += "ls -ltr\n\n"

######output += "csh ./run_GPU_calculation_2devices.csh %s %s %s\n\n" % (tag,data,flat)
output += "csh ./run_GPU_calculation.csh %s %s %s %s\n\n" % (tag,data,flat,flatflag)

output += "ls -ltr\n\n"
output += "pbsdcp -g 'log*dat' $PBS_O_WORKDIR/results/\n"
######output += "pbsdcp -g 'log*log' $PBS_O_WORKDIR/results/\n\n"

pbs_file.write(output)
pbs_file.close()

cmd = ['qsub', pbs_file_name]
sp.Popen(cmd,0).wait()

