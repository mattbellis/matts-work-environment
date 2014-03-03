./read_in_and_fit.py mcstudy_LambdaC_ntp1_pass0/mcstudies_bkg1400_embed_sig50_0064.dat --starting-vals startingValuesForFits/test_gc.txt --baryon LambdaC --ntp ntp1 --pass 0 --fit
./plot_fit_results.py --workspace rootWorkspaceFiles/workspace_default_nfits1.root
./read_in_and_fit.py mcstudy_LambdaC_ntp1_pass0/mcstudies_bkg1400_embed_sig50_0164.dat --starting-vals startingValuesForFits/test_gc.txt --baryon LambdaC --ntp ntp1 --pass 0 --fit --sideband-first --branching-fraction 7.5
./read_in_and_fit_NOCG.py mcstudy_LambdaC_ntp1_pass0/mcstudies_bkg1400_sig20_0164.dat --starting-vals startingValuesForFits/values_for_fits_LambdaC_ntp1_pass0.txt --baryon LambdaC --ntp ntp1 --pass 0 --fit --sideband-first --num-sig 20

python ./read_in_and_fit.py mcstudy_LambdaC_ntp1_pass0/mcstudies_bkg800_embed_sig30_0164.dat --starting-vals startingValuesForFits/test_gc.txt --baryon LambdaC --ntp ntp1 --pass 0 --fit --sideband-first --branching-fraction 7.5
