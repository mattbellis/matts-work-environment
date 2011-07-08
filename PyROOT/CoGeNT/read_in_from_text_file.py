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
    #tbins = 64;
    ############################################################################
    # Define the variables and ranges
    # 
    # Note that the days start at 1 and not 0. 
    ############################################################################
    x = RooRealVar("x","ionization energy (keVee)",0.0,12.0);
    #x = RooRealVar("x","ionization energy (keVee)",0.0,3.0);
    t = RooRealVar("t","time",1.0,tmax+1)
    #t = RooRealVar("t","time",5.0,tmax+5)
    
    myset = RooArgSet()
    myset.add(x)
    myset.add(t)
    data_total = RooDataSet("data_total","data_total",myset)

    x.setRange("sub_x0",0.0,3.0)
    x.setRange("sub_x1",0.5,0.9)
    #x.setRange("sub_x1",1.6,3.0)
    x.setRange("sub_x2",0.5,3.0)
    #x.setRange("sub_x2",0.4,0.5)
    #x.setRange("sub_x2",0.5,1.5)
    x.setRange("sub_x3",0.0,0.4)

    x.setRange("good_days_0",0.5,3.0)
    x.setRange("good_days_1",0.5,3.0)
    x.setRange("good_days_2",0.5,3.0)
    x.setRange("good_days_3",0.5,3.0)

    bin_width = tmax/tbins
    for i in range(0,tbins):
        name = "sub_t%d" % (i)
        lo = i*bin_width + 1;
        hi = (i+1)*bin_width + 1;
        t.setRange(name,lo,hi)

    ############################################################################
    # Dead time Days: 
    # 68-74
    # 102-107
    # 306-308
    ############################################################################
    dead_days = [[68,74], [102,107],[306,308]]
    #dead_days = [[2,100],[390,480]]
    #dead_days = [[100,100]]
    #dead_days = [[60,60], [100,100],[200,200]]
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
            hi = 458+1
        else:
            lo = dead_days[i-1][1]+1
            hi = dead_days[i][0]

        print "%s %d %d" % (name,lo,hi)
        t.setRange(name,lo,hi)


    print fit_range 

    #exit(0)

    ############################################################################
    # Get the pdf
    ############################################################################
    my_pars,sub_pdfs,total_pdf = simple_modulation(t)
    total_pdf.Print("v")

    rrv_dum = RooRealVar("rrv_dum","rrv_dum",10)
    pars_dict = {}
    for p in my_pars:
        if type(p)==type(rrv_dum):
            pars_dict[p.GetName()] = p
            pars_dict[p.GetName()].setConstant(False)



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

            time_days = (t_sec-first_event)/(24.0*3600.0) + 1.0

            '''
            if energy>0.4999 and energy<0.9001:
                print "%f %f" % (time_days, energy)
            '''

            x.setVal(energy)
            t.setVal(time_days)

            if time_days>=68 and time_days<75:
                print time_days
            elif time_days>=102 and time_days<108:
                print time_days
            elif time_days>=306 and time_days<309:
                print time_days

            data_total.add(myset)

            if time_days > 990:
                exit(0);

    #data_total = total_pdf.generate(RooArgSet(x,t),36000)

    ############################################################################
    # Make sure the data is in the live time of the experiment.
    ############################################################################
    data = data_total.reduce(RooFit.CutRange(good_ranges[0]))
    for i in range(1,n_good_spots):
        data.append(data_total.reduce(RooFit.CutRange(good_ranges[i])))

    print "total entries: %d" % (data_total.numEntries())
    print "fit   entries: %d" % (data.numEntries())

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
    #x.setBins(60)
    xframes = []
    for i in xrange(tbins):
        xframes.append([])
        for j in xrange(4):
            xframes[i].append(x.frame(RooFit.Title("Plot of ionization energy")))
            #data.plotOn(xframes[i][j])
            data_reduced_t_x[i][j].plotOn(xframes[i][j])

    xframe_main = x.frame(RooFit.Title("Plot of ionization energy"))
    data.plotOn(xframe_main)

    ############################################################################
    # t
    ############################################################################
    t.setBins(tbins)
    tframes = []
    for i in xrange(4):
        tframes.append(t.frame(RooFit.Title("Plot of ionization energy")))
        data_reduced[i].plotOn(tframes[i])

    ############################################################################
    # Make canvases.
    ############################################################################
    can_x = []
    for i in range(0,4):
        name = "can_x_%s" % (i)
        can_x.append(TCanvas(name,name,10+10*i,10+10*i,1200,900))
        can_x[i].SetFillColor(0)
        can_x[i].Divide(4,4)

    can_x_main = TCanvas("can_x_main","can_x_main",100,100,1200,600)
    can_x_main.SetFillColor(0)
    can_x_main.Divide(1,1)

    can_t = TCanvas("can_t","can_t",200,200,1200,600)
    can_t.SetFillColor(0)
    can_t.Divide(2,2)


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


    can_x_main.cd(1)

    #peak_pars,peak_sub_func,peak_pdf = cosmogenic_peaks(x)
    cogent_pars,cogent_sub_funcs,cogent_energy_pdf = cogent_pdf(x,t)

    # Make a dictionary out of the pars and sub_funcs
    cogent_pars_dict = {}
    for p in cogent_pars:
        cogent_pars_dict[p.GetName()] = p
        cogent_pars_dict[p.GetName()].setConstant(True)

    # 
    cogent_sub_funcs_dict = {}
    for p in cogent_sub_funcs:
        cogent_sub_funcs_dict[p.GetName()] = p

    ############################################################################
    # Try fitting the energy spectrum
    ############################################################################

    cogent_pars_dict["nsig_e"].setVal(4000.0)
    cogent_pars_dict["nsig_e"].setConstant(False)

    cogent_pars_dict["nbkg_e"].setVal(200.0)
    cogent_pars_dict["nbkg_e"].setConstant(False)

    cogent_pars_dict["ncosmogenics_e"].setVal(1013.0)
    cogent_pars_dict["ncosmogenics_e"].setConstant(False)

    cogent_pars_dict["sig_slope"].setVal(-7.5)
    cogent_pars_dict["sig_slope"].setConstant(False)

    #e_fit_range = "%s,%s" % ("sub_x2",fit_range)

    #cogent_energy_pdf.fitTo(data,RooFit.Range("sub_x2"))
    #cogent_energy_pdf.plotOn(xframe_main,RooFit.Range("sub_x2"))

    e_fit_results = cogent_energy_pdf.fitTo(data,
            RooFit.Range(fit_range),
            RooFit.Save(True),
            RooFit.Extended(True)
            )

    #cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(fit_range))
    cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(fit_range), RooFit.NormRange(fit_range))
    #cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(e_fit_range))
    #cogent_energy_pdf.plotOn(xframe_main,RooFit.Range("sub_x2"))

    #'''
    for s in cogent_sub_funcs_dict:
        if "cg_" in s:
            argset = RooArgSet(cogent_sub_funcs_dict[s])
            cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(fit_range),RooFit.Components(argset),RooFit.LineColor(2),RooFit.LineStyle(3))
    for s in cogent_sub_funcs_dict:
        if "_exp" in s:
            argset = RooArgSet(cogent_sub_funcs_dict[s])
            cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(fit_range),RooFit.Components(argset),RooFit.LineColor(36),RooFit.LineStyle(3))
    #'''




    xframe_main.GetXaxis().SetRangeUser(0.0,3.0)
    xframe_main.GetYaxis().SetRangeUser(0.0,200.0)
    xframe_main.Draw()
    gPad.Update()

    fit_results = []
    for i in xrange(4):
        can_t.cd(i+1)

        #if 1:
        if i==1:

            pars_dict["mod_off"].setVal(10.0)
            #pars_dict["mod_off"].setConstant(True)

            pars_dict["nsig"].setVal(644)
            #pars_dict["nsig"].setConstant(True)

            pars_dict["mod_amp"].setVal(3.0)
            #pars_dict["mod_amp"].setConstant(True)

            pars_dict["mod_freq"].setVal(6.28/365.0)
            #pars_dict["mod_freq"].setConstant(True)

            pars_dict["mod_phase"].setVal(-2.5)
            #pars_dict["mod_phase"].setConstant(True)

            fit_result = total_pdf.fitTo(data_reduced[i],
                    RooFit.Save(True),
                    RooFit.Strategy(True),
                    RooFit.Range(fit_range),
                    RooFit.Extended(True)
                    )

            argset = RooArgSet(total_pdf)
            total_pdf.plotOn(tframes[i], RooFit.Components(argset), RooFit.ProjWData(data_reduced[i],True),RooFit.Range(fit_range))
            #total_pdf.plotOn(tframes[i], RooFit.ProjWData(data_reduced[i],True),RooFit.Range(fit_range))
            #total_pdf.plotOn(tframes[i], RooFit.ProjWData(data_reduced[i],True),RooFit.Range("good_days_0,good_days_1,good_days_2,good_days_3"))
            #total_pdf.plotOn(tframes[i], RooFit.ProjWData(data_reduced[i],True),RooFit.Range("good_days_1"))
            fit_result.Print("v")
            fit_results.append(fit_result)

            #temp_data = total_pdf.generate(RooArgSet(t),644)
            #temp_data.plotOn(tframes[i],RooFit.MarkerColor(kRed))

            chi2 = tframes[i].chiSquare()



        tframes[i].Draw()
        gPad.Update()

    for f in fit_results:
        f.Print("v")

    e_fit_results.Print("v")

    print fit_range
    #print e_fit_range

    print "chi2: %f" % (chi2)
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




