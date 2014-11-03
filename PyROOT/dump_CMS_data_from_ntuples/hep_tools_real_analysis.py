import numpy as np
import pickle

################################################################################
################################################################################
def get_collisions(infile,verbose=False):

    collisions = []

    not_at_end = True
    collision_count = 0
    new_collision = True
    while ( not_at_end ):

        ############################################################################
        # Read in one collision
        ############################################################################
        line = infile.readline()

        if collision_count%1000==0 and verbose:
            print "collision count: ",collision_count

        if line=="":
            not_at_end = False

        if line.find("Event")>=0:
            new_collision = True

        if new_collision==True:

            # Read in the jet info for this collision.
            jets = []
            line = infile.readline()
            njets = int(line)
            for i in xrange(njets):
                line = infile.readline()
                vals = line.split()
                e = float(vals[0])
                px = float(vals[1])
                py = float(vals[2])
                pz = float(vals[3])
                bquark_jet_tag = float(vals[4])
                jets.append([e,px,py,pz,bquark_jet_tag])

            # Read in the top jet info for this collision.
            topjets = []
            line = infile.readline()
            ntopjets = int(line)
            for i in xrange(ntopjets):
                line = infile.readline()
                vals = line.split()
                e = float(vals[0])
                px = float(vals[1])
                py = float(vals[2])
                pz = float(vals[3])
                nsub = float(vals[4])
                minmass = float(vals[5])
                topjets.append([e,px,py,pz,nsub,minmass])

            # Read in the muon info for this collision.
            muons = []
            line = infile.readline()
            nmuons = int(line)
            num_mu=0
            for i in xrange(nmuons):
                line = infile.readline()
                vals = line.split()
                e = float(vals[0])
                px = float(vals[1])
                py = float(vals[2])
                pz = float(vals[3])
                #charge = int(vals[4])
                #muons.append([e,px,py,pz,charge])
                muons.append([e,px,py,pz])
                num_mu+=1
                

            # Read in the electron info for this collision.
            electrons = []
            line = infile.readline()
            nelectrons = int(line)
            for i in xrange(nelectrons):
                line = infile.readline()
                vals = line.split()
                e = float(vals[0])
                px = float(vals[1])
                py = float(vals[2])
                pz = float(vals[3])
                #charge = int(vals[4])
                #electrons.append([e,px,py,pz,charge])
                electrons.append([e,px,py,pz])

            # Read in the photon info for this collision.
            '''
            photons = []
            line = infile.readline()
            nphotons = int(line)
            for i in xrange(nphotons):
                line = infile.readline()
                vals = line.split()
                e = float(vals[0])
                px = float(vals[1])
                py = float(vals[2])
                pz = float(vals[3])
                photons.append([e,px,py,pz])
            '''


            # Read in the information about the missing transverse energy (MET) in the collision.
            # This is really the x and y direction for the missing momentum.
            met = []
            line = infile.readline()
            nmet = int(line)
            for i in xrange(nmet):
                line = infile.readline()
                vals = line.split()
                #met_px = float(vals[0])
                #met_py = float(vals[1])
                met_pt = float(vals[0])
                met_phi = float(vals[1])
                met.append([met_pt,met_phi])

            new_collision = False
            collision_count += 1

            collisions.append([jets,topjets,muons,electrons,met])

    return collisions

###############################################################################
'''
def get_collisions_from_zipped_file(infile,verbose=False):
    return get_collisions(infile)
'''

###############################################################################

def get_array_collisions(infile):
    collisions = np.load(infile)
    infile.close()
    return collisions


###############################################################################

def get_compressed_collisions(infile):
    b = np.load(infile)
    collisions = b['arr_0']
    infile.close()
    return collisions


###############################################################################

def get_pickle_collisions(infile):
    collisions = pickle.load(infile)
    infile.close()
    return collisions


###############################################################################
def get_onebyonesixty_collisions(infile):
    toReturn = []
    collisions = get_compressed_collisions(infile)
    for collision in collisions:
        toAdd = []
        tempJets = collision[0:40]
        tempMuons = collision[40:80]
        tempElectrons = collision[80:120]
        tempPhotons = collision[120:160]
        met = collision[160:162].tolist()
        #tempJets = tempJets[tempJets!=0]
        #tempMuons = tempMuons[tempMuons!=0]
        #tempElectrons = tempElectrons[tempElectrons!=0]
        #tempPhotons = tempPhotons[tempPhotons!=0]
        i = 0
        jets = []
        while i+5 <= len(tempJets) and tempJets[i] != 0:
            jets.append(tempJets[i:i+5].tolist())
            i += 5
        muons = []
        i = 0
        while i+5 <=len(tempMuons) and tempMuons[i] != 0:
            muons.append(tempMuons[i:i+5].tolist())
            i += 5
        electrons = []
        i = 0
        while i+5 <= len(tempElectrons) and tempElectrons[i] != 0:
            electrons.append(tempElectrons[i:i+5].tolist())
            i += 5
        photons = []
        i = 0
        while i+5 <= len(tempPhotons) and tempPhotons[i] != 0:
            photons.append(tempPhotons[i:i+4].tolist())
            i += 5
        #jets = np.array(jets)
        #muons = np.array(muons)
        #electrons = np.array(electrons)
        #photons = np.array(photons)
        #met = np.array(met)
        toAdd.append(jets)
        toAdd.append(muons)
        toAdd.append(electrons)
        toAdd.append(photons)
        toAdd.append(met)
        #toAdd = np.array(toAdd)
        toReturn.append(toAdd)
    #toReturn = np.array(toReturn)
    return toReturn

###############################################################################

def get_fourbyforty_collisions(infile):
    toReturn = []
    collisions = get_compressed_collisions(infile)
    for collision in collisions:
        toAdd = []
        tempJets = collision[0][0:40]
        tempMuons = collision[1][0:40]
        tempElectrons = collisions[2][0:40]
        tempPhotons = collision[3][0:40]
        met = collision[3][40:42]
        i = 0
        jets = []
        while i+5 <= len(tempJets) and tempJets[i] != 0:
            jets.append(tempJets[i:i+5].tolist())
            i += 5
        muons = []
        i = 0
        while i+5 <=len(tempMuons) and tempMuons[i] != 0:
            muons.append(tempMuons[i:i+5].tolist())
            i += 5
        electrons = []
        i = 0
        while i+5 <= len(tempElectrons) and tempElectrons[i] != 0:
            electrons.append(tempElectrons[i:i+5].tolist())
            i += 5
        photons = []
        i = 0
        while i+5 <= len(tempPhotons) and tempPhotons[i] != 0:
            photons.append(tempPhotons[i:i+4].tolist())
            i += 5
        toAdd.append(jets)
        toAdd.append(muons)
        toAdd.append(electrons)
        toAdd.append(photons)
        toAdd.append(met)
        #toAdd = np.array(toAdd)
        toReturn.append(toAdd)
    #toReturn = np.array(toReturn)
    return toReturn




###############################################################################

def get_thirtytwobyfive_collisions(infile):
    toReturn = []
    collisions = get_compressed_collisions(infile)
    for collision in collisions:
        toAdd = []
        tempJets = collision[0:8].tolist()
        tempMuons = collision[8:16].tolist()
        tempElectrons = collision[16:24].tolist()
        tempPhotons = collision[24:32].tolist()        
        met = collision[32][0:1].tolist()
        jets = []
        muons = []
        electrons = []
        photons = []
        for jet in tempJets:
            if jet[0] != 0:
                jets.append(jet)
        for muon in tempMuons:
            if muon[0] != 0:
                muons.append(muon)
        for electron in tempElectrons:
            if electron[0] != 0:
                electrons.append(electron)
        for photon in tempPhotons:
            if photon[0] != 0:
                photons.append(photon[0:4])
        
        toAdd.append(jets)
        toAdd.append(muons)
        toAdd.append(electrons)
        toAdd.append(photons)
        toAdd.append(met)
       # toAdd = np.array(toAdd)
        toReturn.append(toAdd)
    #toReturn = np.array(toReturn)
    return toReturn










