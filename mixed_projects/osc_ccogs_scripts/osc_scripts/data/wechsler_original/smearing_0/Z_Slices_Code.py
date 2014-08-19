################################################################################
# This is a very fast way of reading in a text file, when
# you know how the data is formatted, e.g. how many columns
# there are.
# 
# Depending on the size of the file, this still may take some time (~5-20 sec),
# but is still faster than other traditional ways of reading in files.
#
# The trade-off is that this method works best when you have a good amount of 
# memory (RAM) available.
################################################################################



################################################
###                                          ###
###          Import libraries, etc.          ###
###                                          ###
################################################

import numpy as np
import sys
#infile = open('/Users/Chris/Desktop/M_Bellis Research/astro_data/wechsler_gals.cat')
infile_name = sys.argv[1]
infile = open(infile_name)

str_pos = infile_name.find('index')
print str_pos
tag = infile_name[str_pos:str_pos+8]
print tag

################################################
###                                          ###
###    Take the entire file, split it into   ###
###     different values using whitespace    ###
###         (tab, space, end-of-line)        ###
###                                          ###
################################################

content = np.array(infile.read().split()).astype('float')

################################################
###                                          ###
###  A)      How big is this array?          ###
###  B) How many galazies are in this file?  ###
###  C)      Columns are RA, Dec, and Z      ###
###                                          ###
################################################

nentries = len(content)

ncolumns = 3
ngals = nentries/ncolumns
print "# galaxies: %d" % (ngals)

# Now we just need to make an array that has the index of each value we
# want to extract. 
index = np.arange(0,nentries,ncolumns)
# So for three columns, this index array looks like
# [0,3,6,9,12,...,nentries-2]
# We can use this now to pull out the columns we want!
ra =  content[index]
dec = content[index+1]
z =   content[index+2]

# Let's make sure these arrays at least have the same size.
print "\nNumber of entries in coordinate arrays"
print "# ra coords:  %d" % (len(ra))
print "# dec coords: %d" % (len(dec))
print "# z coords:   %d" % (len(z))



################################################
###                                          ###
###  For loop to pull data with slices of Z. ###
###                                          ###
################################################

Zstep = .025
Zmax = .3299888463

for i in range(0,int(Zmax/Zstep)+1):

    output_file_name = "z-slice_%s_%4.3f_to_%4.3f.dat" % (tag,i*Zstep,(i+1)*Zstep)
    output_file = open(output_file_name,'w+')
    
    index2 = z>i*Zstep
    index3 = z<(i+1)*Zstep
    indexT = index2*index3
    #Zslice = z[indexT]
    ra_for_slice = ra[indexT]
    dec_for_slice = dec[indexT]

    #for RA,DEC,Z in zip(ra_for_slice, dec_for_slice, Zslice):
    ngals = len(ra_for_slice)
    output = "%d\n" % (ngals)
    for RA,DEC in zip(ra_for_slice, dec_for_slice):
        output += "%f %f\n" % (RA, DEC)
        
    output_file.write(output)
    output_file.close()

    print "\nZ-array, with values only where Z is greater than "+str(i*Zstep)+" and less than "+str((i+1)*Zstep)

print "Loop finished"
