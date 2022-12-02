#import astropy
from astropy.table import Table
#from astropy.io import fits
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pylab as plt

import os
import sys

#GATHERING THE DATA FROM THE COMPUTER
homedir = os.getenv("HOME")
#print(homedir)
filename = homedir+'/Downloads/'+'all_filament_spines.fits'
filename = sys.argv[1]
#print(filename)

#NAMING THE FILE WE'LL BE TAKING THE DATA FROM
atab = Table.read(filename, format='fits')

#LISTS FROM THE FILE 
filament_ra = (atab['ra'])
filament_dec = (atab['dec'])
filament_name = atab['filament']

# Plot these just for fun!!!!!
# First get the "unique" names for the filaments

################################################################################
def draw_filaments(filament_ra, filament_dec, filament_name, ax=None):
    names = np.unique(filament_name)
    # names is just 1 entry of each name in the filament_name array
    print(names)

    if ax is None:
        plt.figure(figsize=(10,10))
        ax = plt.gca()

    # Loop over the names and pull out the ra and dec for *just that filament*
    # Plot each one!
    for name in names:

        # This mask will select Trues and Falses
        # and the Trues are where the filament_name equals the
        # name that we are looping over at that time.
        mask = filament_name == name
        print(mask)

        # ra and dec for just one filament name
        f_ra =  filament_ra[mask]
        f_dec = filament_dec[mask]

        plt.plot(f_ra, f_dec, 'o', label=name)

    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("filaments.png")
################################################################################



################################################################################
#Function is supposed to take in ra and dec values and output the closest filament to those coordinates
################################################################################
def galaxy_finder(ra,dec, filament_ra, filament_dec, filament_name):
   
   # Initialize this with some massive value
   closest_distance = 1e20
   closest_ra = 0
   closest_dec = 0
   closest_name = None

   # Now, loop over all the points and check their distances

   for f_ra,f_dec,f_n in zip(filament_ra, filament_dec, filament_name):

       distance = np.sqrt((ra - f_ra)**2 + (dec - f_dec)**2)

       # If this distance is closest (so far) then record these values
       if distance < closest_distance:
           closest_ra = f_ra
           closest_dec = f_dec
           closest_name = f_n

           # If this is the closest we found so far, reset the value of closest
           # distance to be *this* one.
           closest_distance = distance
   
   return closest_distance, closest_ra, closest_dec, closest_name   
################################################################################
   

ra = 245
dec = 24

cd, cra, cdec, cname =  galaxy_finder(ra,dec,filament_ra, filament_dec, filament_name)
print(cd, cra, cdec, cname)

# Draw all the filaments
draw_filaments(filament_ra, filament_dec, filament_name)
# Then draw our point and the nearest filament we found
plt.plot(cra,cdec,'kv', markersize=20, label=f'Our point - nearest to {cname}')
plt.legend()
plt.savefig('filaments_with_point.png')

plt.show()
