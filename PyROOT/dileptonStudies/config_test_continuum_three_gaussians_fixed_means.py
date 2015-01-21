def set_initial_values(pars_dict):
    
    pars_dict["frac_12_3_x_offpeak"].setVal(0.10)
    pars_dict["frac_12_3_x_offpeak"].setConstant(False)
    #
    pars_dict["frac_12_x_offpeak"].setVal(0.60)
    pars_dict["frac_12_x_offpeak"].setConstant(False)
    #
    pars_dict["mean_x_offpeak_0"].setVal(0.0)
    pars_dict["mean_x_offpeak_0"].setConstant(True)
    #
    pars_dict["mean_x_offpeak_1"].setVal(0.0)
    pars_dict["mean_x_offpeak_1"].setConstant(True)
    #
    pars_dict["mean_x_offpeak_2"].setVal(0.0)
    pars_dict["mean_x_offpeak_2"].setConstant(True)
    #
    pars_dict["sigma_x_offpeak_0"].setVal(1.0)
    pars_dict["sigma_x_offpeak_0"].setConstant(False)
    #
    pars_dict["sigma_x_offpeak_1"].setVal(3.0)
    pars_dict["sigma_x_offpeak_1"].setConstant(False)
    #
    pars_dict["sigma_x_offpeak_2"].setVal(9.0)
    pars_dict["sigma_x_offpeak_2"].setConstant(False)

