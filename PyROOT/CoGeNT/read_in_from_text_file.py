#/usr/bin/env python

import sys
from ROOT import *

from math import *

from cogent_utilities import *
from cogent_pdfs import *


################################################################################
################################################################################
def main():

    ############################################
    RooMsgService.instance().Print()
    RooMsgService.instance().deleteStream(1)
    RooMsgService.instance().Print()
    ############################################

    first_event = 2750361.2 # seconds

    tmax = 480;
    tbins = 16;
    #tbins = 64;
    t_bin_width = tmax/tbins

    lo_energy = 0.50
    hi_energy = 3.20
    ############################################################################
    # Define the variables and ranges
    # 
    # Note that the days start at 1 and not 0. 
    ############################################################################
    #x = RooRealVar("x","ionization energy (keVee)",0.0,12.0);
    t = RooRealVar("t","time",1.0,tmax+1)
    #x = RooRealVar("x","ionization energy (keVee)",0.0,hi_energy);
    x = RooRealVar("x","ionization energy (keVee)",lo_energy,hi_energy);
    #t = RooRealVar("t","time",5.0,tmax+5)

    x.setRange("FULL",lo_energy,hi_energy)
    t.setRange("FULL",1.0,tmax+1)
    
    myset = RooArgSet()
    myset.add(x)
    myset.add(t)
    data_total = RooDataSet("data_total","data_total",myset)
    data_acc_corr = RooDataSet("data_acc_corr","data_acc_corr",myset)

    x_ranges = [[0.0,hi_energy],
                [lo_energy,0.9],
                [lo_energy,hi_energy],
                [0.0,0.4]
                ]

    x.setRange("good_days_0",lo_energy,hi_energy)
    x.setRange("good_days_1",lo_energy,hi_energy)
    x.setRange("good_days_2",lo_energy,hi_energy)
    x.setRange("good_days_3",lo_energy,hi_energy)

    for i,r in enumerate(x_ranges):

        name = "sub_x%d" % (i)
        x.setRange(name,r[0],r[1])

        for j in range(0,tbins):

            name = "sub_t%d_x%d" % (j,i)
            lo = j*t_bin_width + 1;
            hi = (j+1)*t_bin_width + 1;
            t.setRange(name,lo,hi)

            x.setRange(name,r[0],r[1])

    ############################################################################
    # Dead time Days: 
    # 68-74
    # 102-107
    # 306-308
    ############################################################################
    dead_days = [[68,74], [102,107],[306,308]]
    #dead_days = [[0,1]]
    month_ranges = [[61,90], [91,120],[301,330]]

    #dead_days = [[50,200],[280,320]]
    #dead_days = [[100,100]]
    #dead_days = [[60,60], [100,100],[200,200]]

    acceptance_correction_factor = []
    for d in dead_days:
        acceptance_correction_factor.append(30.0/((30.0-(d[1]-d[0]))))

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

        ########################################
        # Do I need to do this?
        ########################################
        x.setRange(name,lo_energy,hi_energy)


    print "fit_range ---------------------- "
    print fit_range 

    #exit(0)

    ############################################################################
    # Read in from a text file.
    ############################################################################
    infilename = sys.argv[1]
    infile = open(infilename)

    save_file_name = "default"
    if len(sys.argv)>=3:
        save_file_name = sys.argv[2]

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

            # For diagnostics.
            if energy>=lo_energy and energy<=hi_energy:
                data_total.add(myset)

            if time_days > 990:
                exit(0);

    #peak_pars,peak_sub_func,peak_pdf = cosmogenic_peaks(x)
    cogent_pars,cogent_sub_funcs,cogent_energy_pdf = cogent_pdf(x,t)
    print "0--------------"
    cogent_energy_pdf.Print("v")
    print "90--------------"
    #exit()

    # Make a dictionary out of the pars and sub_funcs
    cogent_pars_dict = {}
    for p in cogent_pars:
        if p.GetName().find("gaussian_constraint_")<0:
            cogent_pars_dict[p.GetName()] = p
            cogent_pars_dict[p.GetName()].setConstant(True)

    # 
    cogent_sub_funcs_dict = {}
    for p in cogent_sub_funcs:
        cogent_sub_funcs_dict[p.GetName()] = p

    cogent_pars_dict["nsig_e"].setVal(525.0)
    cogent_pars_dict["nbkg_e"].setVal(800.0)
    #cogent_pars_dict["ncosmogenics"].setVal(681.563)
    #data_total = cogent_energy_pdf.generate(RooArgSet(x,t),2200)

    ############################################################################
    # Make sure the data is in the live time of the experiment.
    ############################################################################
    data = data_total.reduce(RooFit.CutRange(good_ranges[0]))
    print "fit   entries: %d" % (data.numEntries())
    for i in range(1,n_good_spots):
        data.append(data_total.reduce(RooFit.CutRange(good_ranges[i])))
        print "fit   entries: %d" % (data.numEntries())

    print "total entries: %d" % (data_total.numEntries())
    print "fit   entries: %d" % (data.numEntries())

    data_reduced = []
    for i in range(0,4):
        cut_name = "sub_x%d" % (i)
        data_reduced.append(data.reduce(RooFit.CutRange(cut_name)))

    ############################################################################
    # Make frames 
    ############################################################################
    # x
    #x.setBins(240)
    #x.setBins(60)
    #x.setBins(50)
    x.setBins(100)
    xframe_main = x.frame(RooFit.Title("Plot of ionization energy"))
    data.plotOn(xframe_main)

    t.setBins(tbins)
    tframe_main = t.frame(RooFit.Title("Days"))
    data.plotOn(tframe_main)

    hacc_corr = TH1F("hacc_corr","hacc_corr",16,1.0,481)
    nentries = data.numEntries()
    for i in xrange(nentries):
        argset = data.get(i)
        tmp = argset.getRealValue("t")
        correction = 1.0
        for c,m in zip(acceptance_correction_factor,month_ranges):
            if tmp>m[0] and tmp<=m[1]:
                correction = c
                #print "%f %f %f %f" % (tmp,m[0],m[1],correction)

        #print "%f %f %f %f" % (tmp,m[0],m[1],correction)
        hacc_corr.Fill(tmp,correction)

    hacc_corr.SetMarkerSize(0.8)
    hacc_corr.SetMarkerStyle(20)
    hacc_corr.SetMarkerColor(2)
    hacc_corr.SetLineColor(1)


    ############################################################################
    # Make canvases.
    ############################################################################
    can_x_main = TCanvas("can_x_main","can_x_main",100,100,1200,600)
    can_x_main.SetFillColor(0)
    can_x_main.Divide(2,1)

    ############################################################################
    # Try fitting the energy spectrum
    ############################################################################

    cogent_pars_dict["nsig_e"].setVal(525.0)
    cogent_pars_dict["nsig_e"].setConstant(False)

    cogent_pars_dict["nbkg_e"].setVal(700.0)
    cogent_pars_dict["nbkg_e"].setConstant(False)

    #cogent_pars_dict["ncosmogenics"].setVal(681.563)
    #cogent_pars_dict["ncosmogenics"].setConstant(True)

    cogent_pars_dict["sig_slope"].setVal(-4.5)
    cogent_pars_dict["sig_slope"].setConstant(False)

    #cogent_pars_dict["ncosmogenics"].setVal(400)
    #cogent_pars_dict["ncosmogenics"].setConstant(False)

    #cogent_pars_dict["cosmogenic_norms_0"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_1"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_2"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_3"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_4"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_5"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_6"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_7"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_8"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_9"].setConstant(False)
    #cogent_pars_dict["cosmogenic_norms_10"].setConstant(False)



    yearly_mod = 2*pi/365.0
    cogent_pars_dict["sig_mod_frequency"].setVal(yearly_mod); cogent_pars_dict["sig_mod_frequency"].setConstant(True)
    cogent_pars_dict["bkg_mod_frequency"].setVal(yearly_mod); cogent_pars_dict["bkg_mod_frequency"].setConstant(True)

    cogent_pars_dict["sig_mod_phase"].setVal(0.0); cogent_pars_dict["sig_mod_phase"].setConstant(False)
    #cogent_pars_dict["bkg_mod_phase"].setVal(0.0); cogent_pars_dict["bkg_mod_phase"].setConstant(False)
    #cogent_pars_dict["sig_mod_phase"].setVal(0.0); cogent_pars_dict["sig_mod_phase"].setConstant(True)
    cogent_pars_dict["bkg_mod_phase"].setVal(0.0); cogent_pars_dict["bkg_mod_phase"].setConstant(True)

    cogent_pars_dict["sig_mod_amp"].setVal(1.0); cogent_pars_dict["sig_mod_amp"].setConstant(False)
    #cogent_pars_dict["bkg_mod_amp"].setVal(1.0); cogent_pars_dict["bkg_mod_amp"].setConstant(False)
    #cogent_pars_dict["sig_mod_amp"].setVal(0.0); cogent_pars_dict["sig_mod_amp"].setConstant(True)
    cogent_pars_dict["bkg_mod_amp"].setVal(0.0); cogent_pars_dict["bkg_mod_amp"].setConstant(True)

    ############################################################################
    # Construct the RooNLLVar list
    ############################################################################
    print "Creating the NLL variable"
    nllList = RooArgSet()
    temp_list = []
    print nllList
    #nll = None
    print good_ranges
    for i,r in enumerate(good_ranges):
        name = "nll_%s" % (r)
        #nllComp = RooNLLVar(name,name,cogent_energy_pdf,data,RooFit.Extended(True),RooFit.Range(r))
        #nllComp.Print("v")
        #temp_list.append(RooNLLVar(name,name,cogent_energy_pdf,data,RooFit.Extended(True),RooFit.Range(r)))
        #temp_list.append(RooNLLVar(name,name,cogent_energy_pdf,data,RooFit.Range(r)))
        #RooAbsReal* nllComp0 = new RooNLLVar("nll_range0","-log(likelihood)",*total,*data_reduce,kTRUE,"range0",0,1,kFALSE,kFALSE,kFALSE,kFALSE);
        temp_list.append(RooNLLVar(name,name,cogent_energy_pdf,data,True,r,"",1,False,False,False,False))
        print name
        #nllList.add(nllComp)
        nllList.add(temp_list[i])
        print name
        nllList.Print()

    nll = RooAddition("nll","-log(likelihood)",nllList,True);

    m = RooMinuit(nll)

    m.setVerbose(False)
    #m.setStrategy(2)
    m.migrad()
    m.hesse()
    e_fit_results = m.save()

    #cogent_pars_dict["nsig_e"].setVal(4.0*cogent_pars_dict["nsig_e"].getVal())
    #cogent_pars_dict["nbkg_e"].setVal(4.0*cogent_pars_dict["nbkg_e"].getVal())

    ##nll = RooNLLVar("nll","nll",cogent_energy_pdf,data,RooFit.Extended(True),RooFit.Range(fit_range),RooFit.SplitRange(True))
    #nll = RooNLLVar("nll","nll",cogent_energy_pdf,data,RooFit.Extended(True),RooFit.Range(fit_range),RooFit.SplitRange(True))
    ##fit_func = RooFormulaVar("fit_func","nll + log_gc",RooArgList(nll,pars_d["log_gc"]))
    #fit_func = RooFormulaVar("fit_func","nll",RooArgList(nll))
    #m = RooMinuit(fit_func)
    #m.setVerbose(False)
    #m.setStrategy(2)
    #m.migrad()
    #m.hesse()
    #e_fit_results = m.save()

    #exit(1)

    '''
    e_fit_results = cogent_energy_pdf.fitTo(data,
            RooFit.Range(fit_range),
            #RooFit.Extended(True),
            RooFit.Save(True),
            )
    '''

    #fit_range = "FULL"

    #cogent_energy_pdf.plotOn(xframe_main, RooFit.Range(fit_range))
    #cogent_energy_pdf.plotOn(xframe_main)
    #cogent_energy_pdf.plotOn(xframe_main, RooFit.Range(fit_range), RooFit.NormRange(fit_range))
    cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(fit_range),RooFit.NormRange("FULL"))

    rargset = RooArgSet(cogent_energy_pdf)
    cogent_energy_pdf.plotOn(tframe_main, RooFit.Components(rargset), RooFit.Range(fit_range), RooFit.NormRange("FULL"))
    #cogent_energy_pdf.plotOn(tframe_main, RooFit.Components(rargset))
    #cogent_energy_pdf.plotOn(tframe_main, RooFit.Range("FULL"), RooFit.NormRange("FULL"))

    #cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(fit_range), RooFit.NormRange(fit_range))
    #cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(fit_range))
    #cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(e_fit_range))
    #cogent_energy_pdf.plotOn(xframe_main,RooFit.Range("sub_x2"))

    #cogent_energy_pdf.plotOn(tframe_main,RooFit.Range(fit_range), RooFit.NormRange(fit_range))

    #cogent_sub_funcs_dict["cg_total"].plotOn(xframe_main,RooFit.Range(fit_range),RooFit.NormRange(fit_range))

    #'''
    for s in cogent_sub_funcs_dict:
        if "cg_" in s:
            argset = RooArgSet(cogent_sub_funcs_dict[s])
           # cogent_energy_pdf.plotOn(xframe_main,RooFit.Range(fit_range),RooFit.Components(argset),RooFit.LineColor(3),RooFit.LineStyle(2))
            cogent_energy_pdf.plotOn(xframe_main,RooFit.Components(argset),RooFit.LineColor(2),RooFit.LineStyle(2),RooFit.Range(fit_range),RooFit.NormRange("FULL"))
            cogent_energy_pdf.plotOn(tframe_main,RooFit.Components(argset),RooFit.LineColor(2),RooFit.LineStyle(2),RooFit.Range(fit_range),RooFit.NormRange("FULL"))
            #cogent_energy_pdf.plotOn(tframe_main,RooFit.Range(fit_range),RooFit.Components(argset),RooFit.LineColor(2),RooFit.LineStyle(3))
    #'''

    #'''
    for s in cogent_sub_funcs_dict:
        #print s
        if "cg_total" in s:
            #print "Plotting !!!!!!!!!!!!!!!"
            #print s
            argset = RooArgSet(cogent_sub_funcs_dict[s])
            cogent_energy_pdf.plotOn(xframe_main,RooFit.Components(argset),RooFit.LineColor(2),RooFit.LineStyle(1),RooFit.Range(fit_range),RooFit.NormRange("FULL"))
            cogent_energy_pdf.plotOn(tframe_main,RooFit.Components(argset),RooFit.LineColor(2),RooFit.LineStyle(1),RooFit.Range(fit_range),RooFit.NormRange("FULL"))
    #'''


    #'''
    count = 0
    for s in cogent_sub_funcs_dict:
        if "_exp" in s:
            argset = RooArgSet(cogent_sub_funcs_dict[s])
            cogent_energy_pdf.plotOn(xframe_main,RooFit.Components(argset),RooFit.LineColor(26+count),RooFit.LineStyle(1),RooFit.Range(fit_range),RooFit.NormRange("FULL"))
            cogent_energy_pdf.plotOn(tframe_main,RooFit.Components(argset),RooFit.LineColor(26+count),RooFit.LineStyle(1),RooFit.Range(fit_range),RooFit.NormRange("FULL"))
            count += 10
    #'''

    '''
    for s in cogent_sub_funcs_dict:
        if "cosmogenic_pdfs_" in s:
            print s
            argset = RooArgSet(cogent_sub_funcs_dict[s])
            #cogent_sub_funcs_dict[s].plotOn(tframe_main,RooFit.Range(fit_range),RooFit.Components(argset),RooFit.LineColor(2),RooFit.LineStyle(3))
            #cogent_sub_funcs_dict[s].plotOn(xframe_main,RooFit.Components(argset),RooFit.LineColor(2),RooFit.LineStyle(3))
            cogent_sub_funcs_dict[s].plotOn(tframe_main,RooFit.Components(argset),RooFit.LineColor(2),RooFit.LineStyle(3))
    '''


    can_x_main.cd(1)
    #xframe_main.GetXaxis().SetRangeUser(0.0,hi_energy)
    #xframe_main.GetYaxis().SetRangeUser(0.0,200.0)
    xframe_main.GetYaxis().SetRangeUser(0.0,95.0)
    xframe_main.Draw()
    gPad.Update()

    can_x_main.cd(2)
    #tframe_main.GetXaxis().SetRangeUser(0.0,3.0)
    #tframe_main.GetYaxis().SetRangeUser(0.0,200.0)
    tframe_main.Draw()
    hacc_corr.Draw("samee")
    gPad.Update()

    for file_type in ['png','pdf','eps']:
        outfile = "%s.%s" % (save_file_name,file_type)
        can_x_main.SaveAs(outfile)

    cogent_energy_pdf.Print("v")
    e_fit_results.Print("v")
    e_fit_results.correlationMatrix().Print("v")
    print "neg log likelihood: %f" % (e_fit_results.minNll())
    

    #print fit_range
    #print e_fit_range

    #print "\nchi2: %f" % (chi2)
    for i in xrange(4):
        print "%d entries: %d" % (i, data_reduced[i].numEntries())

    days = 0.0
    phase = cogent_pars_dict["sig_mod_phase"].getVal()
    if phase>=0:
        days = 365 - (phase/(2*pi))*365 + (365/4.0)
    else:
        days = (phase/(2*pi))*365 + (365/2.0)
    print "sig phase: %f (rad) %f (days)" % (phase, days)

    phase = cogent_pars_dict["bkg_mod_phase"].getVal()
    if phase>=0:
        days = 365 - (phase/(2*pi))*365 + (365/4.0)
    else:
        days = (phase/(2*pi))*365 + (365/2.0)
    print "bkg phase: %f (rad) %f (days)" % (phase, days)

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




