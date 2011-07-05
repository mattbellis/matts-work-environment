#!/usr/bin/env python

import sys
from ROOT import *

from cogent_utilities import *
from cogent_pdfs import *


################################################################################
################################################################################
def main():

    first_event = 2750361.2 # seconds

    tmax = 480;
    tbins = 16;
    #tbins = 160;
    ############################################################################
    # Define the variables and ranges
    ############################################################################
    x = RooRealVar("x","ionization energy (keVee)",0.0,12.0);
    t = RooRealVar("t","time",0.0,tmax)
    
    myset = RooArgSet()
    myset.add(x)
    myset.add(t)
    data_total = RooDataSet("data_total","data_total",myset)

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
    # Dead time Days: 
    # 68-74
    # 102-107
    # 306-308
    ############################################################################
    dead_days = [[68,74], [102,107],[306,308]]
    n_good_spots = len(dead_days)+1
    good_ranges = []
    fit_range = ""
    for i in range(0,n_good_spots):

        name = "good_days_%d" % (i)
        good_ranges.append(name)
        if i<n_good_spots-1:
            fit_range += "%s," % (name)
        else:
            fit_range += "%s" % (name)

        if i==0:
            lo = 1
            hi = dead_days[i][0]
        elif i==n_good_spots-1:
            lo = dead_days[i-1][1]+1
            hi = tmax
        else:
            lo = dead_days[i-1][1]+1
            hi = dead_days[i][0]

        print "%s %d %d" % (name,lo,hi)
        t.setRange(name,lo,hi)


    print fit_range 

    #exit(0)

    ############################################################################
    # Read in from a text file.
    ############################################################################
    infilename = sys.argv[1]
    infile = open(infilename)

    for line in infile:
        
        vals = line.split()
        if len(vals)==2:
            
            t_sec = float(vals[0])
            amplitude = float(vals[1])

            energy = amp_to_energy(amplitude,0)

            time_days = (t_sec-first_event)/(24.0*3600.0) + 1

            #print "%f %f" % (time_days, energy)

            x.setVal(energy)
            t.setVal(time_days)
            if time_days>=68 and time_days<75:
                print time_days
            elif time_days>=102 and time_days<108:
                print time_days
            elif time_days>=306 and time_days<309:
                print time_days

            data_total.add(myset)

    ############################################################################
    # Make sure the data is in the live time of the experiment.
    ############################################################################
    data = data_total.reduce(RooFit.CutRange(good_ranges[0]))
    for i in range(1,n_good_spots):
        data.append(data_total.reduce(RooFit.CutRange(good_ranges[i])))

    print "total entries: %d" % (data_total.numEntries())
    print "fit   entries: %d" % (data.numEntries())

    #data = total_pdf.generate(RooArgSet(x,t),500) # Gives good agreement with plot
    #data = total_pdf.generate(RooArgSet(x,t),4000)
    data_reduced = []
    for i in range(0,4):
        cut_name = "sub_x%d" % (i)
        data_reduced.append(data.reduce(RooFit.CutRange(cut_name)))

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
        data_reduced[i].plotOn(tframes[i])



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
    for i in range(0,4):
        name = "can_x_%s" % (i)
        can_x.append(TCanvas(name,name,10+10*i,10+10*i,1200,900))
        can_x[i].SetFillColor(0)
        can_x[i].Divide(4,4)

    can_t = TCanvas("can_t","can_t",200,200,1200,600)
    can_t.SetFillColor(0)
    can_t.Divide(2,2)

    my_pars,sub_pdfs,total_pdf = simple_modulation(t)
    total_pdf.Print("v")

    rrv_dum = RooRealVar("rrv_dum","rrv_dum",10)
    pars_dict = {}
    for p in my_pars:
        if type(p)==type(rrv_dum):
            pars_dict[p.GetName()] = p
            pars_dict[p.GetName()].setConstant(False)



    for i in xrange(tbins):
        for j in xrange(4):
            #pad_index = (i%3)*4+(j+1)
            #can_x[i/3].cd(pad_index)
            pad_index = i+1
            can_x[j].cd(pad_index)
            xframes[i][j].GetXaxis().SetRangeUser(0.0,3.0)
            if j==0:
                xframes[i][j].GetYaxis().SetRangeUser(0.0,30.0)
            elif j==1:
                xframes[i][j].GetYaxis().SetRangeUser(0.0,16.0)
            elif j==2:
                xframes[i][j].GetYaxis().SetRangeUser(0.0,20.0)
            xframes[i][j].Draw()
            gPad.Update()



    fit_results = []
    for i in xrange(4):
        can_t.cd(i+1)

        #if 1:
        if i==1:

            pars_dict["mod_off"].setVal(50)
            #pars_dict["mod_off"].setConstant(True)

            pars_dict["nsig"].setVal(644)
            #pars_dict["nsig"].setConstant(True)

            #pars_dict["mod_amp"].setVal(10)
            #pars_dict["mod_amp"].setConstant(True)

            pars_dict["mod_freq"].setVal(6.28/365.0)
            pars_dict["mod_freq"].setConstant(True)

            pars_dict["mod_phase"].setVal(0.0)
            #pars_dict["mod_phase"].setConstant(True)

            fit_result = total_pdf.fitTo(data_reduced[i],
                    RooFit.Save(True),
                    RooFit.Strategy(True),
                    RooFit.Range(fit_range),
                    #RooFit.Range("good_days_1"),
                    #RooFit.Range("good_days_0,good_days_1,good_days_2,good_days_3"),
                    #RooFit.NormRange(fit_range),
                    RooFit.Extended(True))

            #total_pdf.plotOn(tframes[i], RooFit.ProjWData(data_reduced[i],True))
            total_pdf.plotOn(tframes[i], RooFit.ProjWData(data_reduced[i],True),RooFit.Range(fit_range))
            #total_pdf.plotOn(tframes[i], RooFit.ProjWData(data_reduced[i],True),RooFit.Range("good_days_0,good_days_1,good_days_2,good_days_3"))
            #total_pdf.plotOn(tframes[i], RooFit.ProjWData(data_reduced[i],True),RooFit.Range("good_days_1"))
            fit_result.Print("v")
            fit_results.append(fit_result)


        tframes[i].Draw()
        gPad.Update()

    for f in fit_results:
        f.Print("v")

    print fit_range

    print "\n"
    for i in xrange(4):
        print "entries: %d" % (data_reduced[i].numEntries())


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




