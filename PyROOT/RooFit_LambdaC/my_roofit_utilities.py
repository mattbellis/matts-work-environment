from ROOT import *

from array import *

#from backgroundAndSignal_NEW_def import *
#import backgroundAndSignal_NEW_def

################################################################################
################################################################################
################################################################################
#def read_file_return_dataset(infile, x, y, z, psf_lo, psf_hi, dim=3, max_events=1e9, index=0):
def read_file_return_dataset(infile, x, y, z, data_ranges, dim=3, max_events=1e9, index=0):
    nevents=0
    name = "dataset_%d" % (index)
    ds = RooDataSet(name, name, RooArgSet(x,y,z) )

    name = "dataset_z_%d" % (index)
    ds_z = RooDataSet(name, name, RooArgSet(z) )

    print "Data ranges:"
    print data_ranges

    # mES ranges
    x_lo = data_ranges[0][0]
    x_hi = data_ranges[0][1]
    # DeltaE ranges
    y_lo = data_ranges[1][0]
    y_hi = data_ranges[1][1]
    # NN ranges
    z_lo = data_ranges[2][0]
    z_hi = data_ranges[2][1]

    for line in infile.readlines():
        x.setVal(float(line.split()[0]))
        y.setVal(float(line.split()[1]))

        NNval = z_hi - 0.01
        if dim==2:
            NNval = z_hi - 0.01
        else:
            NNval = float(line.split()[2])

        z.setVal(NNval)

        if nevents<max_events:
            #print "%3.3f %3.3f %3.3f" % (x.getVal(), y.getVal(), z.getVal())
            #print "\t%3.3f %3.3f" % (x_lo, x_hi)
            #print "\t%3.3f %3.3f" % (y_lo, y_hi)
            #print "\t%3.3f %3.3f" % (z_lo, z_hi)
            #print "here"

            if z.getVal()>z_lo and z.getVal()<z_hi and  \
               x.getVal()>x_lo and x.getVal()<x_hi and \
               y.getVal()>y_lo and y.getVal()<y_hi:
                #if y.getVal()>y_lo and y.getVal()<y_hi:
                # Run this check otherwise the fit won't converge.
                #print "Adding...."
                ds.add(RooArgSet(x,y,z))
                ds_z.add(RooArgSet(z))
                nevents += 1
                #print "Added event --------------"
            '''
            else:
                print "NOT ADDING EVENT ========+++++++++===========++++++++++++"
            '''


        else:
            #print "hi"
            break

    #print "Here is hte nevents: %d %f" % (nevents, max_events)
    return ds_z, ds


################################################################################
################################################################################
################################################################################
def set_starting_values(fitting_pars, starting_vals):
  # Set the starting values from the file
  for s in starting_vals:
    for p in fitting_pars:
      if s[0] == p.GetName():
        print s[0]
        print s[1]
        print s[2]
        p.setVal(float(s[1]))
        p.setConstant(s[2])
        #print "Setting constant %s %f %s" % ( p.GetName(), p.getVal(), p.isConstant() )

################################################################################
################################################################################
################################################################################
def write_parameters_logfile(my_fit_results, logfilename):
  ###################################################
  # Print the output logfile
  ###################################################
  logfile = open(logfilename, "w+")

  float_pars_final = my_fit_results.floatParsFinal()
  float_pars_init  = my_fit_results.floatParsInit()

  const_pars = my_fit_results.constPars()

  num_float_pars = float_pars_final.getSize()
  num_const_pars = const_pars.getSize()

  #######
  header_output = "%20s%20s%20s%20s\n" % ( "############### Name", "Initial val", "Final val", "Is constant?" )
  logfile.write(header_output)
  for j in range(0,num_float_pars):
    output = "%20s%20.5f%20.5f%20s\n" % ( float_pars_final[j].GetName(), float_pars_init[j].getVal(), float_pars_final[j].getVal(), '0' )
    logfile.write( output )

  #######
  logfile.write( header_output )
  for j in range(0,num_const_pars):
    output = "%20s%20.5f%20.5f%20s\n" % ( const_pars[j].GetName(), const_pars[j].getVal(), const_pars[j].getVal(), '1' )
    logfile.write( output )

  logfile.close()
###################################################
###################################################

################################################################################
################################################################################
def likelihood_curve(fit_results, dataset, scan_points_dataset, fit_func, pdf, dep_var, outfile_name=None):
    frames = []
    gr = TGraph()
    vals = [0.0,0.0,0.0]

    ############################################################################
    RooMsgService.instance().Print()
    RooMsgService.instance().deleteStream(1)
    RooMsgService.instance().Print()
    ############################################################################

    print "IN LIKE CURVE:"
    print pdf

    nll = RooNLLVar("nll","nll", pdf, dataset)
    nll.Print("V")

    dep_var_val = dep_var.getVal()
    dep_var_err = dep_var.getError()
    dep_var_err_hi = dep_var.getAsymErrorHi()
    dep_var_err_lo = dep_var.getAsymErrorLo()

    print "DEP VAR VALS: %f %f" % (dep_var_val,dep_var_err)

    #bestnll = nll.getVal()
    bestnll = fit_func.getVal()
    bestdep_var = dep_var.getVal()
    print "BEST DEP VAR: %f +/- %f" % (bestdep_var, dep_var_err)
    print "BEST LL: %f" % (bestnll)
    print "PDF: %f" % (pdf.getVal())

    #lorange = dep_var_val - 1.0*dep_var_err
    lorange = 0.0 - dep_var_err
    if dep_var_val<0.0:
        lorange = dep_var_val - dep_var_err
    hirange = dep_var_val + 10.0*dep_var_err

    if lorange==0.0 and hirange<0.1:
        lorange = 0.0
        hirange =  1.0

    # Make sure that the hirange can extend to positive numbers.
    # Otherwise, if the best nsig is negative, we will not get a valid
    # result for the upper limit.
    '''
    if bestdep_var<0:
        hirange = 5.0
    '''
    # Just for trying to plot the Lambda0 ntp4
    lorange = -10.0
    hirange = 10.0


    ############################################################################
    # Make a frame with the likelihood curve
    ############################################################################
    print "lorange/hirange: %3.9f %3.9f" % (lorange,hirange)
    frames.append(dep_var.frame(RooFit.Bins(10),RooFit.Range(lorange, hirange), RooFit.Title("-log(L) scan vs nsig"))) # RooPlot
    nll.plotOn(frames[0], RooFit.ShiftToZero(), RooFit.LineColor(kBlue))
    #fit_func.plotOn(frames[0], RooFit.ShiftToZero(), RooFit.LineColor(kBlue))


    ############################################################################
    # Get the area and 90% upper limit
    ################################
    # Calc total area
    ################################
    xpts90 = array('f')
    ypts90 = array('f')
    area = 0.0
    areagreaterthan0 = 0.0
    xpts = array('f')
    ypts = array('f')

    scan_x = array('f')
    scan_y = array('f')
    scan_z = array('f')

    temp_scan_x = []
    temp_scan_y = []
    temp_scan_z = []

    nllpts = array('f')
    bestnllpts = array('f')
    area_gt0 = []

    nll_val_at_0 = 0.0
    nll_diff_at_0 = 0.0
    test_diff = 1e6
    nentries = scan_points_dataset.numEntries()
    print "nentries: %d" % (nentries)
    for i in range(0,nentries):
        #print i
        # We will print out in 10^{-8}
        temp_x = 100.0*scan_points_dataset.get(i).getRealValue("scan_x")
        temp_y = scan_points_dataset.get(i).getRealValue("scan_y")
        temp_z = scan_points_dataset.get(i).getRealValue("scan_z")

        ###############################################################
        # Order the points according to x
        ###############################################################
        if len(scan_x)==0:
            scan_x.append(temp_x)
            scan_y.append(temp_y-bestnll)
            scan_z.append(temp_z)

        #print "len(scan_x): %d" % (len(scan_x))

        for j in range(0,len(scan_x)):
            #print "\t%d" % (j)
            if temp_x>scan_x[j] and len(scan_x)==1 and i>0:
                scan_x.append(temp_x)
                scan_y.append(temp_y-bestnll)
                scan_z.append(temp_z)
                break
            elif temp_x<scan_x[j] and len(scan_x)==1 and i>0:
                scan_x.insert(j,temp_x)
                scan_y.insert(j,temp_y-bestnll)
                scan_z.insert(j,temp_z)
                break
            elif temp_x>scan_x[j] and temp_x<scan_x[j] and i>0:
                scan_x.insert(j+1,temp_x)
                scan_y.insert(j+1,temp_y-bestnll)
                scan_z.insert(j+1,temp_z)
                break
            elif temp_x>scan_x[j] and j==len(scan_x)-1 and i>0:
                scan_x.append(temp_x)
                scan_y.append(temp_y-bestnll)
                scan_z.append(temp_z)
                break
            elif temp_x<scan_x[j] and i>0:
                scan_x.insert(j,temp_x)
                scan_y.insert(j,temp_y-bestnll)
                scan_z.insert(j,temp_z)
                break

        if fabs(temp_x)<test_diff:
            test_diff = fabs(temp_x)
            nll_val_at_0 = temp_y
            nll_diff_at_0 = temp_y-bestnll

    '''
    for x in scan_x:
        print x
    '''

    #master_array = [temp_scan_x,temp_scan_y,temp_scan_z]
    #temp_array = sort(master_array,0)

    ############################################################################
    # Calculate deviation from 0.
    ############################################################################


    # Figure out the appropriate step size
    '''
    step = 0.01
    step = dep_var_err/50.0
    print "STEP SIZE: %f" % (step)
    if bestdep_var<0:
        #step /= 10.0
        lorange = -2.0*dep_var_err
        hirange = 5.0
    '''
    print "num scan pts: %d" % (len(scan_x))
    step = 1
    if len(scan_x)>0:
        step = scan_x[1]-scan_x[0]
    print "step: %f" % (step)
    #n = lorange
    #while n<hirange:
    for i in range(0,nentries):
        #dep_var.setVal(n)
        #val = exp(-(nll.getVal() - bestnll))
        n = scan_x[i]
        val = exp(-(scan_y[i]))
        #print val
        nllpts.append(scan_y[i])
        bestnllpts.append(bestnll)
        area_slice = val*step
        area += area_slice

        if n>0.0:
            xpts90.append(n)
            ypts90.append(val)
            areagreaterthan0 += area_slice
            area_gt0.append(areagreaterthan0)

        xpts.append(n)
        ypts.append(val)
        n += step

        '''
        #if val<0.01 and n>dep_var_val and bestdep_var>0.0:
        if val<0.001 and n>dep_var_val:
            hirange = n
            break
        '''

    ################################
    # Calc 90% area
    ################################
    area90 = 0.0
    hi90 = 0.0
    npts90 = 0
    #total_area_above_0 = area_gt0[-1]
    for k,n in enumerate(xpts90):
        #area90 = area_gt0[k]/area
        if areagreaterthan0>0.0:
            area90 = area_gt0[k]/areagreaterthan0
        else:
            area90 = 0.0
        #print "%4.2f %4.2f %4.2f" % (areagreaterthan0, area_gt0[k], area90)
        npts90 += 1
        if area90>0.9:
            hi90 = n
            break


    '''
    print "Diagnostics"
    print len(xpts)
    print xpts
    print ypts
    '''
    gr_scan = TGraph(nentries, scan_x, scan_y)
    gr_scan.SetName("gr_scan")
    gr_scan.SetMarkerStyle(20)
    gr_scan.SetMarkerSize(0.5)
    gr_scan.SetMarkerColor(2)

    gr = TGraph(len(xpts), xpts, ypts)
    gr.SetName("gr")
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(0.5)
    gr.SetMarkerColor(2)

    '''
    print "Diagnostics"
    print npts90
    print xpts90
    print ypts90
    '''
    gr90 = TGraph()
    if npts90>0:
        gr90 = TGraph(npts90, xpts90, ypts90)
        gr90.SetName("gr90")
        gr90.SetMarkerStyle(20)
        gr90.SetFillStyle(1001)
        gr90.SetFillColor(7)
        gr90.SetMarkerSize(0.5)
        gr90.SetMarkerColor(2)

    graphs = [gr,gr90,gr_scan]
    ul_vals = [area,areagreaterthan0,hi90]

    std_from_0 = [nll_diff_at_0, bestdep_var, dep_var_err]
    print std_from_0

    print "Calculated the area under the llh curve from: %3.3f %3.3f" % (lorange,hirange)
    print "BEST NLL: %6.3f" % (bestnll)
    print "NLL AT 0: %6.3f" % (nll_val_at_0)
    print "NLL DIFF: %6.3f" % (nll_diff_at_0)
    print "Most likely branching fraction: %3.3f" % (std_from_0[1])

    sigma_inconsistent_with_0 = sqrt(2.0*std_from_0[0])
    if bestdep_var<=0.0 or dep_var_err==0.0:
        sigma_inconsistent_with_0 = 0.0

    print "ul (90%s): %3.3f" % ('%',ul_vals[2])
    print "Sigma(inconsistent with 0): %3.3f" % (sigma_inconsistent_with_0)
    print "diff NLL: %3.3f" % (std_from_0[0])
    print "area: %3.3f" % (ul_vals[0])
    print "area greater than 0: %3.3f" % (ul_vals[1])


    # Open an outfile for the scan points, if a file name has
    # been passed in.
    outfile = None
    if outfile_name:
        outfile = open(outfile_name,"w+")
        npts = len(xpts)

        output = "%-30s%3.3f +/- %3.3f   hi/lo ( %f %f )\n" % ("Most_likely_BF: ",std_from_0[1],dep_var_err,dep_var_err_lo,dep_var_err_hi)
        output += "%-30s%3.3f\n" % ("ul_(90%): ",ul_vals[2])
        output += "%-30s%3.3f\n" % ("sigma(inconsistent_with_0): ",sigma_inconsistent_with_0)
        #output += "%-30s%3.3f\n" % ("diff_NLL: ",std_from_0[0])
        #output += "%-30s%3.3f\n" % ("area: ",ul_vals[0])
        #output += "%-30s%3.3f\n" % ("area_greater_than_0: ",ul_vals[1])
        outfile.write(output)
        
        output = ""
        for i in range(0,npts):
            output += "%f %f %f %f %f %f\n" % (xpts[i],ypts[i],nllpts[i],bestnllpts[i],(nllpts[i]-bestnllpts[i]),step)
        outfile.write(output)
        outfile.close()


    return frames[0],graphs,ul_vals,std_from_0

################################################################################
################################################################################
################################################################################
# List of all of the starting input files for data
################################################################################
################################################################################
################################################################################

# LambdaC
# ntp1       B --> Lambda_c+ mu- (SP9446)
# ntp2       B --> Lambda_c+ e-  (SP9445)
#
# Lambda0
# ntpX       B^- --> anti-Lambda^0 mu- (SP9452)
# ntpX       B^- --> anti-Lambda^0 e-  (SP9453)
# ntpX       B^- --> Lambda^0      mu- (SP9454)
# ntpX       B^- --> Lambda^0      e-  (SP9455)


