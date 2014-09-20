# This type of naming scheme
# textFiles/textLambdaC_ntp1_SP9446_newTMVA_cut2.txt

# Structure of this object
# [[starting values tag (sig), nevents to fit to determine starting values], 
#  [starting values tag (bkg), nevents to fit to determine starting values], 
#  fit tag,
#  number of background events to generate for MC studies,
#  [[lo limit on mES, hi limit on mES],[lo limit on DeltaE, hi limit on DeltaE], [lo limit on NN output, hi limit on NN output]],
# pass tag (TMVA_qqbar_all_4vars)
# conversion factor, error
# Dimensionality

def fit_pass_info(baryon="LambdaC", ntp="ntp1", my_pass=0):

    pass_tag = "pass%d" % (my_pass)

    info = [ [0,0], [0,0], 'default']
    # textLambdaC_ntp1_SP9446_newTMVA_cut2.txt
    if baryon=="LambdaC":
        ########################################################################
        if ntp=="ntp1":
            if my_pass==0:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 800, \
                        [5.2,5.3],[-0.2,0.2],[0.66, 1.00], \
                        "TMVA_qqbar_all_6vars", [6.19,1.619],3] # With the correct track/PID pct err.
                        #"TMVA_qqbar_all_6vars", [6.1922,1.612],3]

        ########################################################################
        elif ntp=="ntp2":
            if my_pass==0:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 620, \
                        [5.2,5.3],[-0.2,0.2],[0.78, 1.00], \
                        "TMVA_qqbar_all_6vars", [6.05,1.584],3] # With the correct track/PID pct err.
                        #"TMVA_qqbar_all_6vars", [6.0980,1.576]

    ############################################################################
    ############################################################################
    elif baryon=="Lambda0":

        ########################################################################
        if ntp=="ntp1":
            if my_pass==0:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 450, \
                        [5.22,5.3],[-0.2,0.2],[0.70, 0.99], \
                        "TMVA_qqbar_all_4vars", [94.7831,1.198],3]
            ############
            elif my_pass==1:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 290, \
                        [5.22,5.3],[-0.2,0.2],[0.80, 0.99], \
                        "TMVA_qqbar_all_4vars", [86.38,2.108],2] # With the correct track/PID pct err.
                        #"TMVA_qqbar_all_4vars", [86.38,1.161],2]

        ########################################################################
        elif ntp=="ntp2":
            if my_pass==0:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 430, \
                        [5.2,5.3],[-0.2,0.2],[0.70, 1.02], \
                        "TMVA_qqbar_all_4vars", [96.5885,1.208],3]
            ############
            elif my_pass==1:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 180, \
                        [5.2,5.3],[-0.2,0.2],[0.90, 1.02], \
                        "TMVA_qqbar_all_4vars", [81.76,2.105],2] # With the correct track/PID pct err.
                        #"TMVA_qqbar_all_4vars", [81.76,1.135],2]

        ########################################################################
        elif ntp=="ntp3":
            if my_pass==0:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 370, \
                        [5.2,5.3],[-0.2,0.2],[0.60, 1.00], \
                        "TMVA_qqbar_all_4vars", [106.2172,1.263],3]
            ############
            elif my_pass==1:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 180, \
                        [5.2,5.3],[-0.2,0.2],[0.80, 1.00], \
                        "TMVA_qqbar_all_4vars", [94.29,2.264],2] # With the correct track/PID pct err.
                        #"TMVA_qqbar_all_4vars", [94.29,1.199],2]

        ########################################################################
        elif ntp=="ntp4":
            if my_pass==0:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 120, \
                        [5.2,5.3],[-0.2,0.2],[0.90, 1.01], \
                        "TMVA_qqbar_all_4vars", [97.1903,1.211],3]
            ############
            elif my_pass==1:
                info = [[my_pass, 10000], [my_pass,10000], pass_tag, 80, \
                        [5.2,5.3],[-0.2,0.2],[0.96, 1.01], \
                        "TMVA_qqbar_all_4vars", [90.25,2.185],2] # With the correct track/PID pct err.
                        #"TMVA_qqbar_all_4vars", [90.25,1.181],2]


    return info
