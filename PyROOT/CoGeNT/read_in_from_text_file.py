#!/usr/bin/env python

import sys
from ROOT import *

from cogent_utilities import *


################################################################################
################################################################################
def main():

    tmax = 540;
    tbins = 18;
    ############################################################################
    # Define the variables and ranges
    ############################################################################
    x = RooRealVar("x","ionization energy (keVee)",0.0,12.0);
    t = RooRealVar("t","time",0.0,tmax)
    
    myset = RooArgSet()
    myset.add(x)
    myset.add(t)
    data = RooDataSet("data","data",myset)

    x.setRange("sub_x0",0.0,3.0)
    x.setRange("sub_x1",0.5,0.9)
    #x.setRange("sub_x1",1.6,3.0)
    x.setRange("sub_x2",0.5,3.0)
    x.setRange("sub_x3",0.0,0.5)

    for i in range(0,tbins):
        name = "sub_t%d" % (i)
        lo = i*tmax/tbins;
        hi = (i+1)*tmax/tbins;
        t.setRange(name,lo,hi)

    ############################################################################
    # Read in from a text file.
    ############################################################################
    infilename = sys.argv[1]
    infile = open(infilename)

    first_event = 2750361.2 # seconds

    for line in infile:
        
        vals = line.split()
        if len(vals)==2:
            
            t_sec = float(vals[0])
            amplitude = float(vals[1])

            energy = amp_to_energy(amplitude,0)

            time_days = (t_sec-first_event)/(24.0*3600.0)

            #print "%f %f" % (time_days, energy)

            x.setVal(energy)
            t.setVal(time_days)

            data.add(myset)

    #data = total_pdf.generate(RooArgSet(x,t),500) # Gives good agreement with plot
    #data = total_pdf.generate(RooArgSet(x,t),4000)
    data_reduced0 = data.reduce(RooFit.CutRange("sub_x0"))
    data_reduced1 = data.reduce(RooFit.CutRange("sub_x1"))
    data_reduced2 = data.reduce(RooFit.CutRange("sub_x2"))
    data_reduced3 = data.reduce(RooFit.CutRange("sub_x3"))

    data_reduced_t = []
    data_reduced_t_x = []
    for i in range(0,tbins):
        tname = "sub_t%d" % (i)
        data_reduced_t.append(data.reduce(RooFit.CutRange(tname)))
        data_reduced_t_x.append([])
        data_temp = data.reduce(RooFit.CutRange(tname))
        for j in range(0,4):
            xname = "sub_x%d" % (j)
            #data_reduced_t_x[i].append(data.reduce(RooFit.CutRange(tname),RooFit.CutRange(xname)))
            data_reduced_t_x[i].append(data_temp.reduce(RooFit.CutRange(xname)))




    ############################################################################
    # Make frames 
    ############################################################################
    # x
    x.setBins(240)
    xframes = []
    for i in xrange(tbins):
        xframes.append([])
        for j in xrange(4):
            xframes[i].append(x.frame(RooFit.Title("Plot of ionization energy")))
            #data.plotOn(xframes[i][j])
            data_reduced_t_x[i][j].plotOn(xframes[i][j])

    ############################################################################
    # t
    t.setBins(tbins)
    tframes = []
    for i in xrange(4):
        tframes.append(t.frame(RooFit.Title("Plot of ionization energy")))
        if i==0:
            data_reduced0.plotOn(tframes[i])
        elif i==1:
            data_reduced1.plotOn(tframes[i])
        elif i==2:
            data_reduced2.plotOn(tframes[i])
        elif i==3:
            data_reduced3.plotOn(tframes[i])



    #tot_argset = RooArgSet(total_pdf)
    #total_pdf.plotOn(xframes[0],RooFit.Components(tot_argset),RooFit.LineColor(8),RooFit.ProjWData(data))

    #bkg_argset = RooArgSet(bkg_exp)
    #bkg_argset = RooArgSet(bxg)
    #total_pdf.plotOn(xframes[0],RooFit.Components(bkg_argset),RooFit.LineStyle(2),RooFit.LineColor(4),RooFit.ProjWData(data))

    #sig_argset = RooArgSet(sig_exp)
    #sig_argset = RooArgSet(sxg)
    #total_pdf.plotOn(xframes[0],RooFit.Components(sig_argset),RooFit.LineStyle(2),RooFit.LineColor(2),RooFit.ProjWData(data))

    #lxg_argset = RooArgSet(lxg)
    #total_pdf.plotOn(xframes[0],RooFit.Components(lxg_argset),RooFit.LineStyle(2),RooFit.LineColor(6),RooFit.ProjWData(data))

    ############################################################################
    # Make canvases.
    ############################################################################
    can_x = []
    for i in range(0,int((tbins-1)/3+1)):
        name = "can_x_%s" % (i)
        can_x.append(TCanvas(name,name,10+10*i,10+10*i,1200,900))
        can_x[i].SetFillColor(0)
        can_x[i].Divide(4,4)

    can_t = TCanvas("can_t","can_t",200,200,1200,600)
    can_t.SetFillColor(0)
    can_t.Divide(2,2)

    for i in xrange(tbins):
        for j in xrange(4):
            #pad_index = (i%3)*4+(j+1)
            #can_x[i/3].cd(pad_index)
            pad_index = i+1
            can_x[i/3].cd(pad_index)
            xframes[i][j].GetXaxis().SetRangeUser(0.0,3.0)
            if j==0:
                xframes[i][j].GetYaxis().SetRangeUser(0.0,30.0)
            elif j==1:
                xframes[i][j].GetYaxis().SetRangeUser(0.0,16.0)
            elif j==2:
                xframes[i][j].GetYaxis().SetRangeUser(0.0,20.0)
            xframes[i][j].Draw()
            gPad.Update()

    for i in xrange(4):
        can_t.cd(i+1)
        tframes[i].Draw()
        gPad.Update()

    print "\n"
    print "entries: %d" % (data_reduced0.numEntries())
    print "entries: %d" % (data_reduced1.numEntries())
    print "entries: %d" % (data_reduced2.numEntries())
    print "entries: %d" % (data_reduced3.numEntries())


    ############################################################################
    rep = ''
    while not rep in ['q','Q']:
        rep = raw_input('enter "q" to quit: ')
        if 1<len(rep):
            rep = rep[0]

################################################################################
################################################################################
if __name__ == "__main__":
    main()




