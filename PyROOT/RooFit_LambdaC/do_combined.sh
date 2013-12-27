#!/bin/tcsh 

set choice = $1

if ( $choice == "1" ) then
    python2.5 ./combined_limits.py \
        fit_summary_log_files/workspace_LambdaC_ntp1_unblind_unblinded_data_sig0_bkg900_dim3_nfits1.log \
        fit_summary_log_files/workspace_LambdaC_ntp2_unblind_unblinded_data_sig0_bkg700_dim3_nfits1.log

else if ( $choice == "2" ) then 
    python2.5 ./combined_limits.py \
        fit_summary_log_files/workspace_Lambda0_ntp1_unblind_unblinded_data_sig0_bkg350_dim2_nfits1.log \
        fit_summary_log_files/workspace_Lambda0_ntp2_unblind_unblinded_data_sig0_bkg220_dim2_nfits1.log

else if ( $choice == "3" ) then 
    python2.5 ./combined_limits.py \
        fit_summary_log_files/workspace_Lambda0_ntp3_unblind_unblinded_data_sig0_bkg220_dim2_nfits1.log \
        fit_summary_log_files/workspace_Lambda0_ntp4_unblind_unblinded_data_sig0_bkg80_dim2_nfits1.log

endif
