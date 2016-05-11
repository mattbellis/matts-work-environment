import xml.etree.ElementTree as ET
import matplotlib.pylab as plt
import numpy as np

import sys

def get_info(filename):

    tree = ET.parse(filename)
    root = tree.getroot()

    print root

    for child in root:
        print child.tag, child.attrib

    experiment = root.findall('experiment')[0].text
    print experiment

    year = root.findall('year')[0].text
    print year

    yrescale = float(root.findall('y-rescale')[0].text)
    print yrescale

    xpts = []
    ypts = []
    for d in root.findall('data-values'):
        #print d.text.split()

        d = d.text.replace(';','; ')

        for i,v in enumerate(d.split()):
            #print v
            x = v
            if v.find("{[")>=0:
                x = v.replace("{[","")
                #print x
            if v.find("]}")>=0:
                x = v.replace("]}","")
                #print x
            if v.find(";")>=0:
                x = v.replace(";","")
                #print x
        
            if i%2==0:
                #print x
                xpts.append(float(x))
            else:
                ypts.append(float(x))


    xpts = np.array(xpts)
    ypts = np.array(ypts)

    ypts *= yrescale

    label = "%s %s" % (experiment,year)
    #plt.plot(xpts,ypts,label=label)

    return xpts,ypts,experiment,year

#plt.yscale('log')
#plt.xscale('log')
#plt.legend()
#plt.show()
