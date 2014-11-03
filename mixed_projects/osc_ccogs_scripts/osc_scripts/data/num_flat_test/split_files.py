import numpy as np
import sys

infile_name = sys.argv[1]
infile = open(infile_name)
content = np.array(infile.read().split()).astype('string')
print content

nevents_per_file = int(sys.argv[2])

count = 0
nfiles = 0

nevents = int(content[0])
nfiles = nevents/nevents_per_file
nev_per_file = (nevents_per_file)*np.ones(nfiles)
if nevents%nevents_per_file > 0:
    nfiles += 1
    nev_per_file.append(nevents%nevents_per_file)

print nevents,nfiles,nev_per_file
print nev_per_file[0]

basename = infile_name.split('.dat')[0]
tot_entries = 1
for i,nev in enumerate(nev_per_file):
    outfile_name = "%s_max%d_index%03d.dat" % (basename,nevents_per_file,i)
    print outfile_name
    outfile = open(outfile_name,"w+")
    output = "%d\n" % (nev)
    for j in range(0,2*int(nev),2):
        output += "%s %s\n" % (content[j+tot_entries],content[j+tot_entries+1])
        
    outfile.write(output)
    outfile.close()

    tot_entries += 2*nev # This is because content is split into values, not lines.




