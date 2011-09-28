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
    ############################################################################
    # Define the variables and ranges
    # 
    # Note that the days start at 1 and not 0. 
    ############################################################################
    #x = RooRealVar("x","ionization energy (keVee)",0.0,12.0);
    t = RooRealVar("t","time",1.0,tmax+1)
    #x = RooRealVar("x","ionization energy (keVee)",0.0,3.0);
    x = RooRealVar("x","ionization energy (keVee)",lo_energy,3.0);
    #t = RooRealVar("t","time",5.0,tmax+5)

    x.setRange("FULL",lo_energy,3.0)
    t.setRange("FULL",1.0,tmax+1)
    
    myset = RooArgSet()
    myset.add(x)
    myset.add(t)
    data_total = RooDataSet("data_total","data_total",myset)
    data_acc_corr = RooDataSet("data_acc_corr","data_acc_corr",myset)

    x_ranges = [[0.0,3.0],
                [lo_energy,0.9],
                [lo_energy,3.0],
                [0.0,0.4]
                ]

    x.setRange("good_days_0",lo_energy,3.0)
    x.setRange("good_days_1",lo_energy,3.0)
    x.setRange("good_days_2",lo_energy,3.0)
    x.setRange("good_days_3",lo_energy,3.0)

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
    #dead_days = [[68,74], [102,107],[306,308]]

    dead_days = [[50,200],[280,320]]
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

        ########################################
        # Do I need to do this?
        ########################################
        #x.setRange(name,lo_energy,3.0)


    print "fit_range ---------------------- "
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
            if energy>=lo_energy and energy<=3.0:
                data_total.add(myset)

            if time_days > 990:
                exit(0);

    #peak_pars,peak_sub_func,peak_pdf = cosmogenic_peaks(x)
    cogent_pars,cogent_sub_funcs,cogent_energy_pdf = cogent_pdf(x,t)
    #exit()

    # Make a dictionary out of the pars and sub_funcs
    cogent_pars_dict = {}
    for p in cogent_pars:
        cogent_pars_dict[p.GetName()] = p
        cogent_pars_dict[p.GetName()].setConstant(True)

    # 
    cogent_sub_funcs_dict = {}
    for p in cogent_sub_funcs:
        cogent_sub_funcs_dict[p.GetName()] = p

    cogent_pars_dict["nsig_e"].setVal(525.0)
    cogent_pars_dict["nbkg_e"].setVal(800.0)
    cogent_pars_dict["ncosmogenics_e"].setVal(681.563)
    #data_total = total_pdf.generate(RooArgSet(x,t),2200)
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

    #hacc_corr = TH1F("hacc_corr","hacc_corr",tbins,1,tmax+1)

    data_reduced = []
    for i in range(0,4):
        cut_name = "sub_x%d" % (i)
        data_reduced.append(data.reduce(RooFit.CutRange(cut_name)))

    #data_reduced_t = []
    data_reduced_t_x = []
    for i in range(0,tbins):
        data_reduced_t_x.append([])
        for j in range(0,4):
            tname = "sub_t%d_x%d" % (i,j)
            #data_reduced_t.append(data.reduce(RooFit.CutRange(tname)))
            #data_reduced_t_x.append([])
            data_reduced_t_x[i].append(data.reduce(RooFit.CutRange(tname)))

        ''''
        for j in range(0,4):
            xname = "sub_x%d" % (j)
            #data_reduced_t_x[i].append(data.reduce(RooFit.CutRange(tname),RooFit.CutRange(xname)))
            data_reduced_t_x[i].append(data_temp.reduce(RooFit.CutRange(xname)))
        '''



    ############################################################################
    # Make frames 
    ############################################################################
    # x
    #x.setBins(240)
    #x.setBins(60)
    x.setBins(50)
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

    tframe_main = t.frame(RooFit.Title("Days"))
    data.plotOn(tframe_main)

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
    can_x_main.Divide(2,1)

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



    ############################################################################
    # Try fitting the energy spectrum
    ############################################################################

    cogent_pars_dict["nsig_e"].setVal(525.0)
    cogent_pars_dict["nsig_e"].setConstant(False)

    cogent_pars_dict["nbkg_e"].setVal(800.0)
    cogent_pars_dict["nbkg_e"].setConstant(False)

    cogent_pars_dict["ncosmogenics_e"].setVal(681.563)
    cogent_pars_dict["ncosmogenics_e"].setConstant(True)

    cogent_pars_dict["sig_slope"].setVal(-4.5)
    cogent_pars_dict["sig_slope"].setConstant(False)

    cogent_pars_dict["ncosmogenics_e"].setVal(400)
    cogent_pars_dict["ncosmogenics_e"].setConstant(False)



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


    #e_fit_range = "%s,%s" % ("sub_x2",fit_range)

    #cogent_energy_pdf.fitTo(data,RooFit.Range("sub_x2"))
    #cogent_energy_pdf.plotOn(xframe_main,RooFit.Range("sub_x2"))

    #nll = RooNLLVar("nll","nll",cogent_energy_pdf,data,RooFit.Extended(kTRUE),RooFit.Range(fit_range))
    nll = RooNLLVar("nll","nll",cogent_energy_pdf,data,RooFit.Extended(kTRUE))
    #fit_func = RooFormulaVar("fit_func","nll + log_gc",RooArgList(nll,pars_d["log_gc"]))
    fit_func = RooFormulaVar("fit_func","nll",RooArgList(nll))
    m = RooMinuit(fit_func)
    m.setVerbose(kFALSE)
    m.migrad()
    m.hesse()
    e_fit_results = m.save()


    '''
    e_fit_results = cogent_energy_pdf.fitTo(data,
            RooFit.Range(fit_range),
            RooFit.Extended(True),
            RooFit.Save(True),
            )
    '''

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
        print s
        if "cg_total" in s:
            print "Plotting !!!!!!!!!!!!!!!"
            print s
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
    #xframe_main.GetXaxis().SetRangeUser(0.0,3.0)
    xframe_main.GetYaxis().SetRangeUser(0.0,200.0)
    xframe_main.Draw()
    gPad.Update()

    can_x_main.cd(2)
    #tframe_main.GetXaxis().SetRangeUser(0.0,3.0)
    #tframe_main.GetYaxis().SetRangeUser(0.0,200.0)
    tframe_main.Draw()
    gPad.Update()

    for file_type in ['png','pdf','eps']:
        outfile = "%s.%s" % (save_file_name,file_type)
        can_x_main.SaveAs(outfile)

    # Try plotting on some of the tbins
    '''
    for i in xrange(tbins):
        for j in xrange(4):
            
            if j==2:
                pad_index = i+1
                can_x[j].cd(pad_index)

                sub_fit_range = "sub_t%d_x%d" % (i,j)

                #cogent_energy_pdf.plotOn(xframes[i][j],RooFit.Range(sub_fit_range), RooFit.NormRange(sub_fit_range))
                xframes[i][j].Draw()
                gPad.Update()

    '''



    fit_results = []
    '''
    for i in xrange(4):
        can_t.cd(i+1)

        #if 0:
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

    '''

    '''
    for f in fit_results:
        f.Print("v")
    '''

    cogent_energy_pdf.Print("v")
    e_fit_results.Print("v")
    e_fit_results.correlationMatrix().Print("v")
    print "neg log likelihood: %f" % (e_fit_results.minNll())
    

    #print fit_range
    #print e_fit_range

    #print "\nchi2: %f" % (chi2)
    for i in xrange(4):
        print "%d entries: %d" % (i, data_reduced[i].numEntries())


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




