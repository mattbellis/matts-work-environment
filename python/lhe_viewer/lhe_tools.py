import numpy as np
import sys 

def get_events(infile):

    events = []
    nevents = 0
    newevent = False
    firstline = False
    event = []
    for count,line in enumerate(infile):
        #print count
        #print line

        if line.find('<event>')>=0:
            newevent = True
            firstline = True            
            event = []
            nevents += 1
        elif line.find('</event>')>=0:
            newevent = False
            events.append(event)
        elif newevent:
            vals = line.split()
            #print vals
            if firstline:
                N = int(vals[0]) # Number of particles
                w = float(vals[2]) # Weight
                firstline = False
            else:
                pid = int(vals[0]) # Number of particles
                px,py,pz = np.array(vals[6:9]).astype(float) # Weight
                E = float(vals[9])
                m = float(vals[10])
                s = float(vals[12]) # spin
                particle = [E,px,py,pz,m,pid,s]
                event.append(particle)
        if count>=100000000:
            break

    return events
