import numpy as np
import matplotlib.pylab as plt
import read_in_xml_file
import sys
import seaborn as sn

filenames = ['xml_files/928.xml',
             'xml_files/934.xml',
             'xml_files/689.xml',
             'xml_files/931.xml']

fig = plt.figure(figsize=(10,8))

for f in filenames:

    xpts,ypts,experiment,year = read_in_xml_file.get_info(f)
    label = "%s %s" % (experiment,year)
    plt.plot(xpts,ypts,label=label,linewidth=4)
    #plt.fill_between(xpts,ypts,max(ypts),alpha=0.1,facecolor='yellow')



cogent_filenames = ['upper_limits_0.50-3.2_scans_juan.dat',
             'upper_limits_0.55-3.2_scans_juan.dat',
            'upper_limits_0.50-3.2_scans_nicole.dat',
            'upper_limits_0.55-3.2_scans_nicole.dat'
        ]

cogent_labels = [r'Surf. events param. #1 (E$_{\rm low}$=0.50 keVee)',
          r'Surf. events param. #1 (E$_{\rm low}$=0.55 keVee)',
          r'Surf. events param. #2 (E$_{\rm low}$=0.50 keVee)',
          r'Surf. events param. #2 (E$_{\rm low}$=0.55 keVee)']

cogent_files = []

for i,fn in enumerate(cogent_filenames):

    x,y = np.loadtxt(fn,unpack=True,dtype=float)

    plt.plot(x,y,'-',label=cogent_labels[i],linewidth=4,alpha=0.80)


#plt.yscale('log')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel(r'WIMP-nucleon $\sigma$ [cm$^2$]',fontsize=24)
plt.xlabel(r'WIMP mass [GeV/c$^2$]',fontsize=24)
plt.legend(loc='upper right',fontsize=18)
plt.tight_layout()

plt.yscale('log')
plt.xlim(0,30)
plt.ylim(1e-42,1e-38)
plt.xlabel(r'WIMP mass [GeV/c$^2$]',fontsize=24)
plt.ylabel(r'WIMP-nucleon $\sigma_{\rm SI}$ [cm$^2$]',fontsize=24)
plt.legend(loc='upper right')

plt.savefig('exclusion_plot.png')

plt.show()
