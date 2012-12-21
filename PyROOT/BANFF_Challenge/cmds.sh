nn_fit data.dat bkg.dat sig.dat 0.01
nn_fit BANFF_Challenge/prob2_dataset0.dat BANFF_Challenge/bc2p2bg1mc.dat BANFF_Challenge/bc2p2bg2mc.dat 0.01
nn_fit_for_BANFF_Challenge BANFF_Challenge/prob2_dataset0.dat BANFF_Challenge/bc2p2bg1mc.dat BANFF_Challenge/bc2p2bg2mc.dat BANFF_Challenge/bc2p2sigmc.dat 0.01
nn_fit_for_BANFF_Challenge_MaximumLikelihood prob2_dataset0.dat bc2p2bg1mc.dat bc2p2bg2mc.dat bc2p2sigmc.dat 0.01
nn_fit_for_BANFF_Challenge_MaximumLikelihood default.txt bc2p2bg1mc.dat bc2p2bg2mc.dat bc2p2sigmc.dat 0.00005
nn_fit_for_BANFF_Challenge_MaximumLikelihood default.txt bc2p2bg1mc.dat bc2p2bg2mc.dat bc2p2sigmc.dat 0.00005
nn_fit_for_BANFF_Challenge_MaximumLikelihood default_100.txt bc2p2bg1mc.dat bc2p2bg2mc.dat bc2p2sigmc.dat 0.0005
nn_fit_for_BANFF_Challenge_MaximumLikelihood prob2_dataset8000.dat bc2p2bg1mc.dat bc2p2bg2mc.dat bc2p2sigmc.dat 0.0005
./plot_significances_for_a_study.py logfiles/collated_logfile_nsig0_nbs1000_r0.005.log logfiles/collated_logfile_nsig75_nbs1000_r0.005.log
./plot_significances_for_a_study.py logfiles/collated_logfile_nsig0_nbs1000_r0.005.log logfiles/collated_logfile_nbs1000_r0.005.log
./plot_summaries.py summary_results_data.log data
