#!/usr/bin/env python

# import some modules
import os
import sys
import math
from math import *
from optparse import OptionParser

# For TMVA
from array import array

################################################################################
# Parse out the command line.
#########################################
flags = []
alt_flags = []
cuts_to_text = []
#########################################
parser = OptionParser()
parser.add_option("-t", "--type", dest="type", default="mc", \
        help="Display TYPE of histograms", metavar="TYPE")
parser.add_option("-n", "--ntuplename", dest="ntuplename", default="T", \
        help="Name of the ntuple to grab (ntp1)", metavar="NTUPLENAME")
parser.add_option("-b", "--batch", action="store_true", dest="batch", \
        default=False, help="Run in batch mode and exit")
parser.add_option("-m", "--max",  dest="max", default='1e9', \
        help="Maximum number of events over which to run.")
parser.add_option("-T", "--tag", dest="tag", \
        help="Tag to add on to output files.")
parser.add_option("-p", "--plot_extension", dest="plot_ext", \
        help="Extension to add onto output plots")
parser.add_option("--cuts-to-text", action="append", dest="cuts_to_text", \
        help="Dump some histos to text file, for these specified cuts.")
parser.add_option("-d", "--directory", dest="directory", \
        help="Directory from which to read input root files. This supplements \
        any files on the command line")
parser.add_option("-F", "--flags", action="append", dest="flags", \
        help="Flag to search for in the input root files")
parser.add_option("--alt-flags", action="append", dest="alt_flags", \
        help="File *must* contain at least one of these flags")

# Parse the options
(options, args) = parser.parse_args()

################################################################################
# Parse the command line options so we don't have have to prepend options 
# later on in the code. 
################################################################################

max = 1e9
numntuples = 1
ncuts = 32
type = "mc"
ntuplename = "ntp1"
tag = ""
plot_ext = ""
directory = ""
tmva_classifier_output = [0.0, 0.0]
batchMode = False


if len(sys.argv) <= 1:
  parser.error("Incorrect number of arguments")
else:
  type = options.type
  ntuplename = options.ntuplename
  batchMode = options.batch
  
  if options.directory:
    directory = options.directory
  if options.flags:
    flags = options.flags
  if options.alt_flags:
    alt_flags = options.alt_flags
  if options.max != "":
    max = float(options.max)
  if options.cuts_to_text:
    for c in options.cuts_to_text:
      cuts_to_text.append(int(c))
      print "Will dump cuts to text: " + c

################################################################################
################################################################################
cm2microns = 1e4

numhistos = 24
if (type == "mc"):
  numhistos = 2
elif (type == "kin"):
  numhistos = 4
elif (type == "vtx"):
  numhistos = 30
elif (type == "vtx_compare"):
  numhistos = 16
elif (type == "vtx2D"):
  numhistos = 6

#################
# Remember where we are
#################
home = os.environ["HOME"]
pwd = os.environ["HOME"] + "/PyRoot/dilepTreeHistoCreator/"

import ROOT
from ROOT import *

import myPIDselector
from myPIDselector import *

from color_palette import *

###########################
# TMVA
###########################
#from readTMVAinfo import *

#gStyle.SetOptStat(111111)
gStyle.SetOptStat(0)
set_palette("palette",100)


##########################################
# Import the histos def file
##########################################
histos_to_dump_to_text = []
if type == "mc":
  histos_to_dump_to_text = []
elif type == "kin":
  histos_to_dump_to_text = []
elif type == "vtx":
  histos_to_dump_to_text = []
elif type == "vtx_compare":
  histos_to_dump_to_text = []
elif type == "vtx2D":
  histos_to_dump_to_text = []

################################
################################
cuts = []
for c in range(0,128):
  cuts.append(True)
################################


pps = PIDselector("proton")
kps = PIDselector("k")
pips = PIDselector("pi")
mups = PIDselector("mu")
eps = PIDselector("e")

#########################################
#########################################
from histos_def import *

#print "%s %d %d %d" % (type, numntuples, numhistos, ncuts)
h = myHistos(type, numntuples, numhistos, ncuts)
#########################################
#########################################

##########################################
# Create a chain based on the ntuple names
##########################################
print "Printing args:"
print args
ngoodfiles = 0
filenames = []
#ntuplename = sys.argv[1]
t = []
for i in range(0, numntuples):
    t.append(TChain(ntuplename))
    ##########################################
    # Read in the files over which you want to loop
    ##########################################
    for j in args:
        filename = j
        goodFile = True
        for f in flags:
            if filename.find(f) < 0:
                goodFile = False
        if goodFile == True:
            print("Adding file: " + filename)
            t[i].Add(filename)
            filenames.append(filename)

if directory != "":
  
    print directory
    dirList=os.listdir(directory)

    for filename in dirList:
        goodFile = True
        for f in flags:
            if filename.find(f) < 0:
                goodFile = False

        # Search for at least one of the alt flags.
        foundOneAltFlag = False
        if len(alt_flags)==0:
            foundOneAltFlag = True # no alt flags passed in.

        for a in alt_flags:
            if filename.find(a)>=0:
                foundOneAltFlag = True

        if not foundOneAltFlag:
            goodFile = False

        if goodFile == True:
            filename = "%s/%s" % (directory,filename)
            print("Adding file: " + filename)
            t[i].Add(filename)
            filenames.append(filename)

##########################################
##########################################

######################################################
# If we will be dumping to text, create the directory
# and files.
######################################################
textout = []
textindex = []
if cuts_to_text:
    for c in cuts_to_text:

        outname = "%s/textOutput/text_%s_%s_TMVA_%s_%s_%svars%s_cut%s.txt" % \
            (pwd,baryon,ntuplename,options.tmva_background_samples,\
            options.tmva_sample_size,options.tmva_num_vars,options.tag,str(c))

        print "Opening " + outname
        textout.append(open( outname , "w+"))
        textindex.append(c)

######################################################
######################################################
cut_text = []
################################
pstart = 12
kstart = 24
pistart = 10
mustart = 16
estart = 6

################################
Bvtx_mc = [TVector3(), TVector3()]
vtx_diff_mc = TVector3()

Bvtx = [TVector3(), TVector3()]
vtx_diff = TVector3()

Bvtx_softPi = [TVector3(), TVector3()]
vtx_diff_softPi = TVector3()

Bvtx_softPi_tf = [TVector3(), TVector3()]
vtx_diff_softPi_tf = TVector3()

################################

#for i in range(0, numntuples):
i = 0
t = [TTree()]
print filenames
for fn in filenames:
    print "Opening: %s" % (fn)
    infile = TFile(fn)
    t[i] = infile.Get(ntuplename)

    # disable/enable certain branches to increase the speed
    print "Setting branches..."
    t[i].SetBranchStatus("*",1)

    #t[i].SetBranchStatus("p1cm",1)
    #t[i].SetBranchStatus("oldVtx1",1)
    #t[i].SetBranchStatus("oldVtx2",1)
    #t[i].SetBranchStatus("softPi_and_lepton_1_vtx",1)
    #t[i].SetBranchStatus("softPi_and_lepton_2_vtx",1)

    t[i]._p1cm  = TLorentzVector()
    t[i].SetBranchAddress( 'p1cm', t[i]._p1cm )
    t[i].GetBranch('p1cm').GetAddress()

    t[i]._oldVtx1  = TLorentzVector()
    t[i].SetBranchAddress( 'oldVtx1', t[i]._oldVtx1 )
    t[i].GetBranch('oldVtx1').GetAddress()

    t[i]._oldVtx2  = TLorentzVector()
    t[i].SetBranchAddress( 'oldVtx2', t[i]._oldVtx2 )
    t[i].GetBranch('oldVtx2').GetAddress()

    t[i]._softPi_and_lepton_1_vtx  = TVector3()
    t[i].SetBranchAddress( 'softPi_and_lepton_1_vtx', t[i]._softPi_and_lepton_1_vtx )
    t[i].GetBranch('softPi_and_lepton_1_vtx').GetAddress()

    t[i]._softPi_and_lepton_2_vtx  = TVector3()
    t[i].SetBranchAddress( 'softPi_and_lepton_2_vtx', t[i]._softPi_and_lepton_2_vtx )
    t[i].GetBranch('softPi_and_lepton_2_vtx').GetAddress()

    t[i]._softPi_and_lepton_1_vtx_tf  = TVector3()
    t[i].SetBranchAddress( 'softPi_and_lepton_1_vtx_tf', t[i]._softPi_and_lepton_1_vtx_tf )
    t[i].GetBranch('softPi_and_lepton_1_vtx_tf').GetAddress()

    t[i]._softPi_and_lepton_2_vtx_tf  = TVector3()
    t[i].SetBranchAddress( 'softPi_and_lepton_2_vtx_tf', t[i]._softPi_and_lepton_2_vtx_tf )
    t[i].GetBranch('softPi_and_lepton_2_vtx_tf').GetAddress()

    #t[i]._trkp4  = []
    #t[i].SetBranchAddress( 'trkp4', t[i]._trkp4 )
    #t[i].GetBranch('trkp4').GetAddress()
    t[i].SetBranchStatus('trkp4',1)

    #t[i].SetBranchStatus('softPi1_trkidx',1)
    #t[i].SetBranchStatus('softPi2_trkidx',1)
    #t[i].SetBranchStatus('mass2Nu1',1)
    #t[i].SetBranchStatus('mass2Nu2',1)
    #t[i].SetBranchStatus('mcLund',1)
    #t[i].SetBranchStatus('mothIdx',1)
    #t[i].SetBranchStatus('mcvtx',1)
    #t[i].SetBranchStatus('trkmcidx',1)




    # event loop
    nentries = t[i].GetEntries()
    print"Entries: %d"  % (nentries)
    if max < nentries:
        nentries = int(max) 

    # Allow to start at something other than 0. 
    for n in range(0,nentries):

        if n % 1000 == 0:
          #print "Event number",n
          print "Event number " + str(n) + " out of " + str(nentries)

        t[i].GetEntry(n)


        pbit = 0
        kbit = 0
        pibit = 0

        '''
        if t[i].LambdaCd1Idx[ t[i].Bd1Idx[0] ] < 20:
            if t[i].pTrkIdx[ t[i].LambdaCd1Idx[ t[i].Bd1Idx[0] ] ] < 20:
                pbit = t[i].pSelectorsMap[ t[i].pTrkIdx[ t[i].LambdaCd1Idx[ t[i].Bd1Idx[0] ] ] ]
        '''

        lepton_p3 = 0.0
        lepton_costh = 0.0
        lepton_bit = 0
        '''
        lepton_bit = t[i].muSelectorsMap[ t[i].muTrkIdx[ t[i].Bd2Idx[0] ] ]
        lepton_p3 =    t[i].mup3[ t[i].Bd2Idx[0] ] 
        lepton_costh = t[i].mucosth[ t[i].Bd2Idx[0] ]
        '''

        '''
        pps.SetBits(pbit)
        kps.SetBits(kbit)
        pips.SetBits(pibit)

        pcut = pps.IsBitSet(0 + pstart)
        kcut = kps.IsBitSet(0 + kstart)
        picut = pips.IsBitSet(0 + pistart)
        pcut = pps.IsBitSet(3 + pstart)
        kcut = kps.IsBitSet(1 + kstart)
        picut = pips.IsBitSet(1 + pistart)
        '''

        #####################################################
        # Try the dummy cuts
        #####################################################
        dummy_cut = False
        #lund_cut = t[i].BLund[0]>0.0

        truth_matched = True
        #truth_matched = t[i].LambdaCMCIdx[ t[i].Bd1Idx[0] ]>=0
        #electron_truth_matched = t[i].ebrMCIdx[t[i].Bd2Idx[0]]>=0
        #print electron_truth_matched
        #########################

        first_time = True
        soft_pi_cut = False


        #####################################################
        if type=="vtx" or type=='vtx2D' or type=='vtx_compare':
            Bvtx_mc[0] = getattr(t[i],'mcvtx')[3]
            Bvtx_mc[1] = getattr(t[i],'mcvtx')[4]
            Bvtx_mc[0] *= cm2microns
            Bvtx_mc[1] *= cm2microns

            Bvtx[0] = t[i].oldVtx1
            Bvtx[1] = t[i].oldVtx2
            Bvtx[0] *= cm2microns
            Bvtx[1] *= cm2microns

            Bvtx_softPi[0] = t[i].softPi_and_lepton_1_vtx
            Bvtx_softPi[1] = t[i].softPi_and_lepton_2_vtx
            Bvtx_softPi[0] *= cm2microns
            Bvtx_softPi[1] *= cm2microns

            Bvtx_softPi_tf[0] = t[i].softPi_and_lepton_1_vtx_tf
            Bvtx_softPi_tf[1] = t[i].softPi_and_lepton_2_vtx_tf
            Bvtx_softPi_tf[0] *= cm2microns
            Bvtx_softPi_tf[1] *= cm2microns

            vtx_diff_mc.SetXYZ(Bvtx_mc[0].X(),Bvtx_mc[0].Y(),Bvtx_mc[0].Z())
            vtx_diff_mc -= Bvtx_mc[1]

            vtx_diff.SetXYZ(Bvtx[0].X(),Bvtx[0].Y(),Bvtx[0].Z())
            vtx_diff -= Bvtx[1]

            vtx_diff_softPi.SetXYZ(Bvtx_softPi[0].X(),Bvtx_softPi[0].Y(),Bvtx_softPi[0].Z())
            vtx_diff_softPi -= Bvtx_softPi[1]
            if vtx_diff_softPi.Mag()==0.0:
                vtx_diff_softPi.SetXYZ(-1e6,-1e6,-1e6);

            vtx_diff_softPi_tf.SetXYZ(Bvtx_softPi_tf[0].X(),Bvtx_softPi_tf[0].Y(),Bvtx_softPi_tf[0].Z())
            vtx_diff_softPi_tf -= Bvtx_softPi_tf[1]
            if vtx_diff_softPi_tf.Mag()==0.0:
                vtx_diff_softPi_tf.SetXYZ(-1e6,-1e6,-1e6);

            softPi1_trkidx = t[i].softPi1_trkidx;
            softPi2_trkidx = t[i].softPi2_trkidx;
            mass2Nu1 = t[i].mass2Nu1;
            mass2Nu2 = t[i].mass2Nu2;

            p1pmag = -1000.0
            p2pmag = -1000.0
            #'''
            #print "ntrks: %d" % (t[i].ntrks)
            if softPi1_trkidx>=0:
                #print "pi1 trkidx: %d" % (softPi1_trkidx)
                p1pmag = t[i].trkp4[softPi1_trkidx].Rho()
                #print "pi1 pmag: %f" % (p1pmag)
            if softPi2_trkidx>=0:
                #print "pi2 trkidx: %d" % (softPi2_trkidx)
                p2pmag = t[i].trkp4[softPi2_trkidx].Rho()
                #print "pi2 pmag: %f" % (p2pmag)
            #'''

            #soft_pi_cut = p1pmag<0.3 and p2pmag<0.3
            soft_pi_cut = mass2Nu1<3.5 and mass2Nu2<3.5

            ####################################################################
            # Look for the D*'s in the truth tables
            #
            # lep 1 will be positive, so dstar1 [0 here] will be negative, 
            # as will the soft pion.
            ####################################################################
            mcLund = []
            mothIdx = []
            for j in xrange(len(t[i].mcLund)):
                mcLund.append(t[i].mcLund[j])

            for j in xrange(len(t[i].mothIdx)):
                mothIdx.append(t[i].mothIdx[j])
            
            # Is there a dstar in the generation
            dstar_truth = [False,False]
            dstar_charge = [-1, +1]
            for j,c in enumerate(dstar_charge):

                dstar_lund = 413
                B_lund = -511

                dstar_idx = -1
                dstar_lund *= c
                B_lund *= c

                #print "lunds: %d %d %d" % (dstar_lund,B_lund,c)
                if dstar_lund in mcLund:
                    dstar_idx = mcLund.index(dstar_lund)
                    Bmoth_idx = mothIdx[dstar_idx]
                    moth_lund = mcLund[Bmoth_idx] 

                    # Mixing or no mixing
                    moth_lund = abs(moth_lund)
                    B_lund = abs(B_lund)

                    if moth_lund == B_lund:
                        #print "FOUND A D* EVENT! ---- %d" % (c)
                        dstar_truth[j] = True

            # Is the lepton correctly identified
            lep_truth =  [False,False]
            lep_trk_idx = [t[i].p1trkidx, t[i].p2trkidx]
            for j,idx in enumerate(lep_trk_idx):
                mcidx = t[i].trkmcidx[idx]
                # is it an electron or muon
                if abs(mcLund[mcidx])==11 or abs(mcLund[mcidx])==13:
                    mothidx = mothIdx[mcidx]
                    # Did it come from a B
                    if abs(mcLund[mothidx])==511:
                        lep_truth[j] = True
                        #print "Truth matched a lepton from a B decay! %d" % (j)

            # Is the soft pion correctly identified
            sp_truth = [False,False]
            sp_trk_idx = [t[i].softPi1_trkidx, t[i].softPi2_trkidx]
            for j,idx in enumerate(sp_trk_idx):
                mcidx = t[i].trkmcidx[idx]
                # is it a pion
                if abs(mcLund[mcidx])==211:
                    mothidx = mothIdx[mcidx]
                    # Did it come from a D*
                    if abs(mcLund[mothidx])==413:
                        mothidx = mothIdx[mothidx]
                        # Did the D* come from a B
                        if abs(mcLund[mothidx])==511:
                            sp_truth[j] = True
                            #print "Truth matched a pion from a D* from a B decay! %d" % (j)

            ####################################################################

        #####################################################
        # 0 out the cuts 
        ncuts = 3
        for c in range(0,ncuts):
            cuts[c] = True
        #####################################################

        for c in range(0, ncuts):
            if c==0:
                cuts[c] = True
                if first_time:
                    cut_text.append("No cuts")
            elif c==1:
                cuts[c] = cuts[c-1] and soft_pi_cut and dstar_truth[0] and dstar_truth[1]
                if first_time:
                    cut_text.append("Soft pi cut")
            elif c==2:
                cuts[c] = cuts[c-1] and sp_truth[0] and sp_truth[1] and lep_truth[0] and lep_truth[1]
                if first_time:
                    cut_text.append("Soft pion and lepton truth matched")

            # Fill the histos
            if cuts[c]:
                if type == "mc":
                    h[i][0][c].Fill(0.1)
                    h[i][1][c].Fill(-0.1)

                elif type == "kin":
                    h[i][0][c].Fill(t[i].p1cm.Rho())

                elif type == "vtx_compare":
                    h[i][0][c].Fill(vtx_diff_mc.Z())
                    h[i][1][c].Fill(vtx_diff.Z())
                    h[i][2][c].Fill(vtx_diff_softPi.Z())
                    h[i][3][c].Fill(vtx_diff_softPi_tf.Z())
                    h[i][4][c].Fill(vtx_diff_mc.Mag())
                    h[i][5][c].Fill(vtx_diff.Mag())
                    h[i][6][c].Fill(vtx_diff_softPi.Mag())
                    h[i][7][c].Fill(vtx_diff_softPi_tf.Mag())
                    h[i][8][c].Fill(mass2Nu1)
                    h[i][9][c].Fill(mass2Nu2)
                    h[i][10][c].Fill(p1pmag)
                    h[i][11][c].Fill(p2pmag)
                    h[i][12][c].Fill(vtx_diff.Z() - vtx_diff_mc.Z())
                    h[i][13][c].Fill(vtx_diff_softPi.Z() - vtx_diff_mc.Z())
                    h[i][14][c].Fill(vtx_diff_softPi_tf.Z() - vtx_diff_mc.Z())
                    h[i][15][c].Fill(vtx_diff_softPi_tf.Z() - vtx_diff_softPi.Z())

                elif type == "vtx":
                    h[i][0][c].Fill(Bvtx_mc[0].X())
                    h[i][1][c].Fill(Bvtx_mc[0].Y())
                    h[i][2][c].Fill(Bvtx_mc[0].Z())
                    h[i][3][c].Fill(Bvtx_mc[0].Mag())
                    h[i][4][c].Fill(Bvtx_mc[0].Perp())

                    h[i][5][c].Fill(Bvtx_mc[1].X())
                    h[i][6][c].Fill(Bvtx_mc[1].Y())
                    h[i][7][c].Fill(Bvtx_mc[1].Z())
                    h[i][8][c].Fill(Bvtx_mc[1].Mag())
                    h[i][9][c].Fill(Bvtx_mc[1].Perp())

                    h[i][10][c].Fill(vtx_diff_mc.X()*1e4)
                    h[i][11][c].Fill(vtx_diff_mc.Y())
                    h[i][12][c].Fill(vtx_diff_mc.Z())
                    h[i][13][c].Fill(vtx_diff_mc.Mag())
                    h[i][14][c].Fill(vtx_diff_mc.Perp())

                    h[i][15][c].Fill(Bvtx[0].X())
                    h[i][16][c].Fill(Bvtx[0].Y())
                    h[i][17][c].Fill(Bvtx[0].Z())
                    h[i][18][c].Fill(Bvtx[0].Mag())
                    h[i][19][c].Fill(Bvtx[0].Perp())

                    h[i][20][c].Fill(Bvtx[1].X())
                    h[i][21][c].Fill(Bvtx[1].Y())
                    h[i][22][c].Fill(Bvtx[1].Z())
                    h[i][23][c].Fill(Bvtx[1].Mag())
                    h[i][24][c].Fill(Bvtx[1].Perp())

                    h[i][25][c].Fill(vtx_diff.X())
                    h[i][26][c].Fill(vtx_diff.Y())
                    h[i][27][c].Fill(vtx_diff.Z())
                    h[i][28][c].Fill(vtx_diff.Mag())
                    h[i][29][c].Fill(vtx_diff.Perp())

                elif type=="vtx2D":
                    #print "%f %f" % (Bvtx_mc[0].X(),Bvtx_mc[0].Y())
                    h[i][0][c].Fill(Bvtx_mc[0].X(),Bvtx_mc[0].Y())
                    h[i][1][c].Fill(Bvtx_mc[1].X(),Bvtx_mc[1].Y())
                    h[i][2][c].Fill(Bvtx[0].X(),Bvtx[0].Y())
                    h[i][3][c].Fill(Bvtx[1].X(),Bvtx[1].Y())

                    h[i][4][c].Fill(vtx_diff_mc.X(),vtx_diff_mc.Y())
                    #h[i][4][c].Fill(Bvtx_mc[0].X()-Bvtx_mc[1].X(),Bvtx_mc[0].Y()-Bvtx_mc[1].Y())
                    h[i][5][c].Fill(vtx_diff.X(),   vtx_diff.Y())


            ######################################################
            # Dump to text if necessary
            ######################################################
            if cuts_to_text:
                if c in cuts_to_text:
                    index = cuts_to_text.index(c)
                    output = ""
                    #output += str(t[i].LambdaC_unc_Mass[0]) + " "
                    output += str(bpostfitmes) + " "
                    output += "\n"
                    textout[index].write( output )

        first_time = False

        
################################################################################
################################################################################
# Finished looping over all the events.
################################################################################
      
if cuts_to_text:
    for j in range(0,len(textout)):
        textout[j].close()
        print "Closing textfile...."
        print textout[j]
        
################################################################################
################################################################################
# Make the canvases
################################################################################
################################################################################
if not cuts_to_text:

    text = []
    legend = []
    can = []
    for i in range(0, numntuples):
        text.append([])
        can.append([])
        legend.append([])

        ######################################################
        # For the non 2d stuff
        ######################################################
        if type=="mc" or type=="kin" or type=="vtx" or type=="vtx_compare":
            for j in range(0, numhistos):
                name = "can" + str(i) + "_" + str(j)
                candum = TCanvas(name, name, 10+10*j, 10+10*j, 600, 400)
                can[i].append(candum)
                can[i][j].SetFillColor(0)
                can[i][j].Divide(1,1)

                can[i][j].cd(1)
                gPad.SetFillColor(0)
                gPad.SetBorderSize(0)
                gPad.SetRightMargin(0.10)
                gPad.SetLeftMargin(0.15)
                gPad.SetBottomMargin(0.15)


                ###############################
                #  Draw the histos
                ###############################
                h[i][j][0].Draw("")
                h[i][j][0].Draw("samee")
                for k in range(1, ncuts):
                    h[i][j][k].Draw("same")
                    h[i][j][k].Draw("samee")

                ###############################
                # Legend
                ###############################
                legdum = TLegend(0.75, 0.75, 0.99, 0.99)
                for k in range(0, ncuts):
                    num0 = float(h[i][j][0].Integral())
                    num  = float(h[i][j][k].Integral())
                    words =  "Entries: %d" % (num0)

                    if k!=0 and num0!=0:
                        words =  "%s %2.1f" % ("%", 100*num/num0)

                    legdum.AddEntry(h[i][j][k], words, "f")

                legend[i].append(legdum)
                legend[i][j].Draw()

                gPad.Update()

                if options.plot_ext:
                    name = "%s/LeptBc_bellis/Plots/bnvTreeHistoCreator/Plots/%s_%s_%s.%s" % (home, can[i][j].GetName(), type, options.tag, options.plot_ext)
                    can[i][j].SaveAs(name)


        ######################################################
        # For the 2d stuff
        ######################################################
        else:
            print "printing 2D %d" % (ncuts)
            for j in range(0, numhistos):
                can[i].append([])
                legend[i].append([])

                max_events = 0
                for k in range(0, ncuts):
                    name = "can" + str(i) + "_" + str(j) + "_" + str(k)
                    candum = TCanvas(name, name, 10 + 100*j + 50*k, 10+50*k, 600, 400)
                    can[i][j].append(candum)
                    can[i][j][k].SetFillColor(0)
                    can[i][j][k].Divide(1,1)

                    can[i][j][k].cd(1)
                    gPad.SetFillColor(0)
                    gPad.SetBorderSize(0)
                    gPad.SetRightMargin(0.20)
                    gPad.SetLeftMargin(0.20)
                    gPad.SetBottomMargin(0.15)

                    ###############################
                    #  Draw the histos
                    ###############################
                    h[i][j][k].Draw("colz")
                
                    ###############################
                    # Legend
                    ###############################
                    legdum = TLegend(0.75, 0.90, 0.99, 0.99)
                    num0 = float(h[i][j][k].Integral())
                    words =  "Entries: %d" % (num0)
                    legdum.AddEntry(h[i][j][k], words, "")

                    legend[i][j].append(legdum)
                    legend[i][j][k].Draw()

                    gPad.Update()

                    if options.plot_ext:
                        name = "%s/LeptBc_bellis/Plots/bnvTreeHistoCreator/Plots/%s_%s_%s.%s" % (home, can[i][j].GetName(), type, options.tag, options.plot_ext)
                        can[i][j].SaveAs(name)

else:

    outname = "%s/cut_percent_logs/text_%s_%s_TMVA_%s_%s_%svars%s_cut%s.txt" % \
        (pwd,baryon,ntuplename,options.tmva_background_samples,\
        options.tmva_sample_size,options.tmva_num_vars,options.tag,str(c))

    print "Opening " + outname
    textcutout =open( outname , "w+")

################################################################################
################################################################################

    # Dump the stats on the cuts
    for i in range(0, numntuples):
        j=1
        num0 = 0.0
        for k in range(0, ncuts):
            prev_pct = 100.0
            prev = num0
            num0 = float(h[i][j][k].Integral())
            if k==0:
                max_events = num0
            else:
                if prev!=0:
                    prev_pct = 100.0*(num0/prev)
                else:
                    prev_pct = 0.0


            pct = 100.0*(num0/max_events)
            output = "%-21s & %8d & %5.2f & %5.2f \\\\\n" % (cut_text[k], num0, pct, prev_pct)
            print output
            textcutout.write(output)
    textcutout.close()

################################################################################
################################################################################

############################ 
# Save the histos
############################
#if options.rootfilename!=None:
if 1:
    rname = "%s/rootFiles/%s_%s_%s.root" % \
          (pwd,ntuplename,type,options.tag)
    rfile=TFile(rname,"recreate")
    print "Saving to file %s" % (rname)
    for i in range(0, numntuples):
        for j in range(0, numhistos):
            for k in range(0, ncuts):
                h[i][j][k].Write()
    rfile.Write()
    rfile.Close()
    print "Wrote and closed %s" % (rname)



## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
    if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q' ]:
            rep = raw_input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]
                                                                                                                                                                                            

# python style to define the main function
#if __name__ == "__main__":
  #ee()
  ####sys.exit(0)
