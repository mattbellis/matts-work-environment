#!/usr/bin/env python
import sys
from autoread import ntuplereader
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *
#from ROOT import RooFit, RooRealVar, RooGaussian, RooDataSet, RooArgList, RooTreeData
#from ROOT import RooCmdArg, RooArgSet, kFALSE, RooLinkedList
#from ROOT import gStyle
from math import sqrt

def defaultHistoSettings(h):
    h.SetNdivisions(6)
    h.GetYaxis().SetTitleSize(0.09)
    h.GetYaxis().SetTitleFont(42)
    h.GetYaxis().SetTitleOffset(0.7)
    h.GetYaxis().CenterTitle()
    h.GetYaxis().SetNdivisions(6)
#    h.GetYaxis().SetTitle("events/MeV")
    h.SetFillColor(9)

def defaultPadSettings():
    
    gPad.SetFillColor(0)
    gPad.SetBorderSize(0)
    gPad.SetRightMargin(0.20);
    gPad.SetLeftMargin(0.20);
    gPad.SetBottomMargin(0.15);

def waitForInput():
    rep = ''
    while not rep in [ 'c', 'C' ]:
        rep = raw_input( 'enter "c" to continue: ' )
        if 1 < len(rep):
            rep = rep[0]

def maxBinContent(histogram,numBins):
    maxContent = -9999999
    for bin in range(0,numBins):
        content = histogram.GetBinContent(bin)
        if content>maxContent:
            maxContent = content
    return maxContent


# mES sideband 5.27>mES>5.2 (or should I use 5.22)

#inputFiles = ["onpeakr1_ddv3.root","onpeakr2_ddv3.root","onpeakr3_ddv3.root","onpeakr4_ddv3.root","onpeakr5_ddv3.root","onpeakr6_ddv3.root"]
inputFiles = ["onpeakr2_ddv3.root"]
print inputFiles

#inputFiles = ["onpeakr5_ddv3.root"]

variables = ["thetaPrime", "m0Prime", "TFlv", "TCat", "mes"]

#numEventsToGenerate = 400000
numEventsToGenerate = 1000

#outFileName = "offpeak.dev.root"
outFileName = "offpeak.dev.root"

print "Opening input files: onpeakr*_ddv3.root"

#Hist2dXNumBins = 100
#Hist2dYNumBins = 100

Hist2dXNumBins = 50
Hist2dYNumBins = 50

Hist2dXLower = 0.0
Hist2dXUpper = 1.0
Hist2dYLower = 0.0
Hist2dYUpper = 1.0

# sidebandLower = 5.22
sidebandLower = 5.20
sidebandUpper = 5.27

dalitzRegionNames = [\
"Low",\
"LowMiddle_Low",\
"LowMiddle_Middle",\
"LowMiddle_High",\
"MiddleHigh_Low",\
"MiddleHigh_Middle",\
"MiddleHigh_High",\
"High"\
]

dalitzRegionLimits = [\
[[0.0,0.2],[0.0,1.0]],\
[[0.2,0.7],[0.0,0.2]],\
[[0.2,0.7],[0.2,0.8]],\
[[0.2,0.7],[0.8,1.0]],\
[[0.7,0.85],[0.0,0.1]],\
[[0.7,0.85],[0.1,0.9]],\
[[0.7,0.85],[0.9,1.0]],\
[[0.85,1.0],[0.0,1.0]]\
]




#dalitzRegionNames = [\
#"Low",\
#"High"\
#]
#
#dalitzRegionLimits = [\
#[[0.0,0.5],[0.0,1.0]],\
#[[0.5,1.0],[0.0,1.0]]\
#]

#dalitzRegionNames = [\
#"All"\
#]
#
#dalitzRegionLimits = [\
#[[0.0,1.0],[0.0,1.0]]\
#]




regionEventCount = [[[0 for k in range(2)] for j in range(8)] for i in range(len(dalitzRegionLimits))]

# B Background PDFs for storage in btocharmless_keys2d.root:
#
# Positions in the array correspond to different types of B tags (for pos and neg tags resp):
# 0 - All Tags
# 1 - Lepton (TCat 63)
# 2 - Kaon1 (TCat 64)
# 3 - Kaon2 (TCat 65)
# 4 - KaonPion (TCat 66)
# 5 - Pion (TCat 67)
# 6 - Other (TCat 68)
# 7 - No Tag (TCat 0)
# RooRealVar[Taging category][flavor tag (0=pos 1=neg)][variable]

m0Prime = RooRealVar("m0Prime","m0^{'}",Hist2dXLower,Hist2dXUpper)
thetaPrime = RooRealVar("thetaPrime","#Theta^{'}",Hist2dYLower,Hist2dYUpper)

dp_vars = RooArgSet(m0Prime,thetaPrime)

#for i in range(len(dalitzRegionLimits)):
#    regionName = dalitzRegionNames[i]
#    xmin = dalitzRegionLimits[i][0][0]
#    xmax = dalitzRegionLimits[i][0][1]
#    ymin = dalitzRegionLimits[i][1][0]
#    ymax = dalitzRegionLimits[i][1][1]
#    m0Prime.setRange(regionName, xmin, xmax)
#    thetaPrime.setRange(regionName, ymin, ymax)

#
# RooDataSet[Tagging category][taging flavor]
#	RooDataSet*** dp_data = new (RooDataSet**)[8][2];

dp_data = [[[0 for k in range(2)] for j in range(8)] for i in range(len(dalitzRegionLimits))]

for regionNum in range(0,len(dalitzRegionLimits)):
    rns = "_" + str(regionNum)
    dp_data[regionNum][0][0] = RooDataSet("pos_all_tags_dp_Data"+rns,"pos_all_tags_dp_Data"+rns,dp_vars)
    dp_data[regionNum][0][1] = RooDataSet("neg_all_tags_dp_Data"+rns,"neg_all_tags_dp_Data"+rns,dp_vars)
#
    dp_data[regionNum][1][0] = RooDataSet("pos_lepton_tag_dp_Data"+rns,"pos_lepton_tag_dp_Data"+rns,dp_vars)
    dp_data[regionNum][1][1] = RooDataSet("neg_lepton_tag_dp_Data"+rns,"neg_lepton_tag_dp_Data"+rns,dp_vars)
#
    dp_data[regionNum][2][0] = RooDataSet("pos_kaon1_tag_dp_Data"+rns,"pos_kaon1_tag_dp_Data"+rns,dp_vars)
    dp_data[regionNum][2][1] = RooDataSet("neg_kaon1_tag_dp_Data"+rns,"neg_kaon1_tag_dp_Data"+rns,dp_vars)
#
    dp_data[regionNum][3][0] = RooDataSet("pos_kaon2_tag_dp_Data"+rns,"pos_kaon2_tag_dp_Data"+rns,dp_vars)
    dp_data[regionNum][3][1] = RooDataSet("neg_kaon2_tag_dp_Data"+rns,"neg_kaon2_tag_dp_Data"+rns,dp_vars)
#
    dp_data[regionNum][4][0] = RooDataSet("pos_kaonpion_tag_dp_Data"+rns,"pos_kaonpion_tag_dp_Data"+rns,dp_vars)
    dp_data[regionNum][4][1] = RooDataSet("neg_kaonpion_tag_dp_Data"+rns,"neg_kaonpion_tag_dp_Data"+rns,dp_vars)
#
    dp_data[regionNum][5][0] = RooDataSet("pos_pion_tag_dp_Data"+rns,"pos_pion_tag_dp_Data"+rns,dp_vars)
    dp_data[regionNum][5][1] = RooDataSet("neg_pion_tag_dp_Data"+rns,"neg_pion_tag_dp_Data"+rns,dp_vars)
#
    dp_data[regionNum][6][0] = RooDataSet("pos_other_tag_dp_Data"+rns,"pos_other_tag_dp_Data"+rns,dp_vars)
    dp_data[regionNum][6][1] = RooDataSet("neg_other_tag_dp_Data"+rns,"neg_other_tag_dp_Data"+rns,dp_vars)
#
    dp_data[regionNum][7][0] = RooDataSet("pos_notag_tag_dp_Data"+rns,"pos_notag_tag_dp_Data"+rns,dp_vars)
    dp_data[regionNum][7][1] = RooDataSet("neg_notag_tag_dp_Data"+rns,"neg_notag_tag_dp_Data"+rns,dp_vars)


ntuple = ntuplereader(inputFiles,"ntp",variables)
nTotal = ntuple.getEntries()

print "Processing", nTotal, "events"

thetaVal = 0
m0Val = 0
TFlvVal = 0
TCatVal = 0
mesVal = 0

good2DEntries = 0

for entryNum in range(0,nTotal):
#for i in range(0,5000):
#for i in range(0,100000):
               
    ntuple.entry(entryNum)
    
    thetaVal = ntuple.get("thetaPrime")
    m0Val = ntuple.get("m0Prime")
    TFlvVal = ntuple.get("TFlv")
    TCatVal = ntuple.get("TCat")
    mesVal = ntuple.get("mes")
           
    if (m0Val>-99 and thetaVal>-99 and m0Val >= Hist2dXLower and m0Val <= Hist2dXUpper and thetaVal >= Hist2dYLower and thetaVal <= Hist2dYUpper and TCatVal > -99 and TFlvVal > -99 and mesVal >= sidebandLower and mesVal <= sidebandUpper):
               
    #############################
    #Is this the proper usage of TFlv? Why is it never 0 even though TCat can be untagged ?
    #############################

        flavor = 0
        if(TFlvVal>0.0):
            flavor = 0
        elif(TFlvVal<0.0):
            flavor = 1
        else:
            print "Error: TFlv==0"

        tagType = -1

        # tagType 0 corresponds to all tags combined

        tagDict = {63 : 1,\
                   64 : 2,\
                   65 : 3,\
                   66 : 4,\
                   67 : 5,\
                   68 : 6,\
                   0  : 7}

        tagType = tagDict.get(TCatVal)

        m0Prime.setVal(m0Val)
        thetaPrime.setVal(thetaVal)

        # x-axis var is m0Prime

        curRegion = -1
        for regionNum in range(0,len(dalitzRegionLimits)):
            xmin = dalitzRegionLimits[regionNum][0][0]
            xmax = dalitzRegionLimits[regionNum][0][1]
            ymin = dalitzRegionLimits[regionNum][1][0]
            ymax = dalitzRegionLimits[regionNum][1][1]

            if m0Val>=xmin and m0Val < xmax and thetaVal>=ymin and thetaVal<ymax:
                curRegion = regionNum

        dp_data[curRegion][tagType][flavor].add(dp_vars)
        dp_data[curRegion][0][flavor].add(dp_vars)

        regionEventCount[curRegion][tagType][flavor] += 1
        regionEventCount[curRegion][0][flavor] += 1

        good2DEntries += 1

print good2DEntries, " entries were added to each 2d histogram"
    
dp_PDF_Array = [[[0,0] for k in range(0,1)] for regionNum in range(0,len(dalitzRegionLimits))]

print "================================================"
print "======   Generating smoothed RooKeysPDFs  ======"
print "================================================"


for j in range(0,1):
    for k in range(0,1):
        for regionNum in range(0,len(dalitzRegionLimits)):
            print "\tGenerating dp_PDF_Array[" + str(regionNum) + "][" + str(j) + "][" + str(k) + "]........."
            name = "dp_PDF_Array_%d_%d_%d" % (j, k, regionNum)
            dp_PDF_Array[regionNum][j][k] = Roo2DKeysPdf(name,name,m0Prime,thetaPrime,dp_data[regionNum][j][k],"m",1.)
            #dp_PDF_Array[regionNum][j][k] = Roo2DKeysPdf("dp_PDF","dp_PDF",m0Prime,thetaPrime,dp_data[regionNum][j][k],"m",1.)

            print "regionEventCount[" + str(regionNum) + "][" + str(j) + "][" + str(k) + "] = " + str(regionEventCount[regionNum][j][k])

# The Roo2DKeys options available are:
#      a = select an adaptove bandwidth [default]
#      n = select a normal bandwidth
#      m = mirror kernal contributions at edges [fold gaussians back into the x,y plane]
#      d = print debug statements [useful for development only; default is off]
#      v  = print verbose debug statements [useful for development only; default is off]
#      vv = print ludicrously verbose debug statements [useful for development only; default is off]
 
names = [[["Smooth_pos_Tag_Cont_DP"+"_"+str(regionNum), "Smooth_neg_Tag_Cont_DP"+"_"+str(regionNum)],\
         ["Smooth_pos_Tag_Cont_DP_Lepton"+"_"+str(regionNum), "Smooth_neg_Tag_Cont_DP_Lepton"+"_"+str(regionNum)],\
         ["Smooth_pos_Tag_Cont_DP_Kaon1"+"_"+str(regionNum), "Smooth_neg_Tag_Cont_DP_Kaon1"+"_"+str(regionNum)],\
         ["Smooth_pos_Tag_Cont_DP_Kaon2"+"_"+str(regionNum), "Smooth_neg_Tag_Cont_DP_Kaon2"+"_"+str(regionNum)],\
         ["Smooth_pos_Tag_Cont_DP_KaonPion"+"_"+str(regionNum), "Smooth_neg_Tag_Cont_DP_KaonPion"+"_"+str(regionNum)],\
         ["Smooth_pos_Tag_Cont_DP_Pion"+"_"+str(regionNum), "Smooth_neg_Tag_Cont_DP_Pion"+"_"+str(regionNum)],\
         ["Smooth_pos_Tag_Cont_DP_Other"+"_"+str(regionNum), "Smooth_neg_Tag_Cont_DP_Other"+"_"+str(regionNum)],\
         ["Smooth_pos_Tag_Cont_DP_NoTag"+"_"+str(regionNum), "Smooth_neg_Tag_Cont_DP_NoTag"+"_"+str(regionNum)]]\
for regionNum in range(len(dalitzRegionLimits))]


regionlessNames = [["Smooth_pos_Tag_Cont_DP", "Smooth_neg_Tag_Cont_DP"],\
         ["Smooth_pos_Tag_Cont_DP_Lepton", "Smooth_neg_Tag_Cont_DP_Lepton"],\
         ["Smooth_pos_Tag_Cont_DP_Kaon1", "Smooth_neg_Tag_Cont_DP_Kaon1"],\
         ["Smooth_pos_Tag_Cont_DP_Kaon2", "Smooth_neg_Tag_Cont_DP_Kaon2"],\
         ["Smooth_pos_Tag_Cont_DP_KaonPion", "Smooth_neg_Tag_Cont_DP_KaonPion"],\
         ["Smooth_pos_Tag_Cont_DP_Pion", "Smooth_neg_Tag_Cont_DP_Pion"],\
         ["Smooth_pos_Tag_Cont_DP_Other", "Smooth_neg_Tag_Cont_DP_Other"],\
         ["Smooth_pos_Tag_Cont_DP_NoTag", "Smooth_neg_Tag_Cont_DP_NoTag"]]
    
outFile2D = TFile(outFileName,"UPDATE")

DP_ROODATASET_Array = [[[0,0] for j in range(0,1)] for regionNum in range(len(dalitzRegionLimits))]

totalEvents = [[0 for j in range(1)] for i in range(1)]
# The first tagging category corresponds to all tagging categories
for tagCat in range(0,1):
    for tagFlav in range(0,1):
        for regionNum in range(0,len(dalitzRegionLimits)):
            totalEvents[tagCat][tagFlav] += regionEventCount[regionNum][tagCat][tagFlav]

generatedEvents = [[[0 for k in range(1)] for j in range(1)] for i in range(len(dalitzRegionLimits))]
for tagCat in range(0,1):
    for tagFlav in range(0,1):
        for regionNum in range(0,len(dalitzRegionLimits)):
            if totalEvents[tagCat][tagFlav] != 0:                
                generatedEvents[regionNum][tagCat][tagFlav] = regionEventCount[regionNum][tagCat][tagFlav] / (totalEvents[tagCat][tagFlav]+0.0) * numEventsToGenerate
            else:
                print "\ttotalEvents[" + str(tagCat) + "][" + str(tagFlav) + "] == 0"

print "================================================"
print "======   Generating histograms from PDFs  ======"
print "================================================"


cachedPDF = []
for tagCat in range(0,1):
  cachedPDF.append([])
  for tagFlav in range(0,1):
    cachedPDF[tagCat].append([])
    #for regionNum in range(0, 2):
    for regionNum in range(0,len(dalitzRegionLimits)):
      print "\tFilling DP_ROODATASET_Array[" + str(regionNum) + "][" + str(tagCat) + "][" + str(tagFlav) + "] from dp_PDF_ARRAY[regionNum][tagCat][tagFlav]"
      eventsToGenerate = int(round(generatedEvents[regionNum][tagCat][tagFlav]))

      print "\t\tGenerating", eventsToGenerate, "events for region", regionNum

#            m0Prime.setBins(Hist2dXNumBins,"cache")
#            thetaPrime.setBins(Hist2dYNumBins,"cache")

      m0Prime.setBins(Hist2dXNumBins)
      thetaPrime.setBins(Hist2dYNumBins)

      name = "mycachedPDF%d_%d_%d" % (tagCat, tagFlav, regionNum)
      print name
      cachedPDF[tagCat][tagFlav].append(RooCachedPdf(name, name, dp_PDF_Array[regionNum][tagCat][tagFlav], RooArgSet(m0Prime,thetaPrime)))
      #cachedPDF = RooCachedPdf("cachedPDF","cachedPDF", dp_PDF_Array[regionNum][tagCat][tagFlav])

      # Try this
      #cachedPDF[tagCat][tagFlav][regionNum].fillCacheObject()
      #cachedPDF[tagCat][tagFlav][regionNum]
      print "Here"
      print cachedPDF[tagCat][tagFlav][regionNum].getVal( RooArgSet(m0Prime, thetaPrime) )
      print cachedPDF[tagCat][tagFlav][regionNum].getCachePdf( RooArgSet(m0Prime, thetaPrime) )

      

      #DP_ROODATASET_Array[regionNum][tagCat][tagFlav] = cachedPDF.generate(RooArgSet(m0Prime,thetaPrime), eventsToGenerate)
      #DP_ROODATASET_Array[regionNum][tagCat][tagFlav] = dp_PDF_Array[regionNum][tagCat][tagFlav].generate(RooArgSet(m0Prime,thetaPrime), eventsToGenerate)
      DP_ROODATASET_Array[regionNum][tagCat][tagFlav] = cachedPDF[tagCat][tagFlav][regionNum].generate(RooArgSet(m0Prime,thetaPrime), eventsToGenerate)


#            DP_ROODATASET_Array[regionNum][tagCat][tagFlav] = dp_PDF_Array[regionNum][tagCat][tagFlav].generate(RooArgSet(m0Prime,thetaPrime), eventsToGenerate)

#            DP_ROODATASET_Array[regionNum][tagCat][tagFlav] = dp_PDF_Array[regionNum][tagCat][tagFlav].generate(RooArgSet(m0Prime,thetaPrime), eventsToGenerate, RooFit.Range(dalitzRegionNames[regionNum]))

print "========================================================="
print "======   Filling regionless histograms from pdfs   ======"
print "========================================================="

DP_PDF_Histo_Array = [[0,0] for j in range(0,1)]
for tagCat in range(0,1):
    for tagFlav in range(0,1):
        DP_PDF_Histo_Array[tagCat][tagFlav] = TH2F(regionlessNames[tagCat][tagFlav],regionlessNames[tagCat][tagFlav],Hist2dXNumBins,Hist2dXLower,Hist2dXUpper,Hist2dYNumBins,Hist2dYLower,Hist2dYUpper)




        #for regionNum in range(0, 2):
        for regionNum in range(len(dalitzRegionLimits)):            
            curRooDataset = DP_ROODATASET_Array[regionNum][tagCat][tagFlav]

            eventsInDataset = curRooDataset.numEntries()

            print "Adding", eventsInDataset, "events to DP_PDF_Histo_Array[", tagCat, "][", tagFlav, "] from region", regionNum
            
            curRooDataset.fillHistogram(DP_PDF_Histo_Array[tagCat][tagFlav],RooArgList(m0Prime,thetaPrime))



#        fullDataSet = RooDataSet("fullDataSet","fullDataSet",RooArgSet(m0Prime,thetaPrime))
#        for regionNum in range(0,len(dalitzRegionLimits)):
#            eventsInDataset = DP_ROODATASET_Array[regionNum][tagCat][tagFlav].numEntries()
#            print "Adding", eventsInDataset, "events to fullDataSet[", tagCat, "][", tagFlav, "] from region", regionNum
#
#            fullDataSet.append(DP_ROODATASET_Array[regionNum][tagCat][tagFlav])
#
#        fullDataSet.fillHistogram(DP_PDF_Histo_Array[tagCat][tagFlav],RooArgList(m0Prime,thetaPrime))




#dp_PDF_Array[regionNum][tagCat][tagFlav].createHistogram(names[regionNum][tagCat][tagFlav], m0Prime, RooFit.Binning(Hist2dXNumBins, Hist2dXLower, Hist2dXUpper),RooFit.YVar(thetaPrime,RooFit.Binning(Hist2dYNumBins, Hist2dYLower, Hist2dYUpper)))

        DP_PDF_Histo_Array[tagCat][tagFlav].Write(regionlessNames[tagCat][tagFlav])

outFile2D.Write()
outFile2D.Close()
