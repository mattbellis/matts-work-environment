import numpy as np
import sys

################################################################################
# Get all the data
################################################################################
infile_name = sys.argv[1]
infile = open(infile_name)
content = np.array(infile.read().split()).astype('float')

nentries = len(content)
ncolumns = 3
ngals = nentries/ncolumns

index = np.arange(0,nentries,ncolumns)
ra =  content[index]
dec = content[index+1]
z =   content[index+2]

print "Read in all the values...."

ra_arcsec = ra*3600
dec_arcsec = dec*3600

print "Converted degrees to arc seconds...."

basename = infile_name.split('.cat')[0]
n_datasets_start = 10
n_datasets_end = 20
if len(sys.argv)>2:
    n_datasets_start = int(sys.argv[2])
    n_datasets_end = int(sys.argv[3])

for i in xrange(n_datasets_start,n_datasets_end):
    outfile_name = "smearing_0/%s_index%03d.dat" % (basename,i)
    print outfile_name

    # Smear the z values using a Gaussian of with a width of 10% of z.
    # IS THIS ABS REALLY WHAT I SHOULD BE USING HERE?
    sigma = np.abs(0.03*(1+z))
    new_z = np.random.normal(z,sigma)
    print "Smeared the the z-values..."

    outfile = open(outfile_name,"w+")
    output = ""
    for ir,id,iz in zip(ra_arcsec,dec_arcsec,new_z):
        output += "%.3f %.3f %f\n" % (ir,id,iz)
        
    outfile.write(output)
    outfile.close()
