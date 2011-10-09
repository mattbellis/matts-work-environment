#!/bin/tcsh 


################################################################################
# Do some studies to look at the different errors for the Gaussian constraint
################################################################################
foreach flag(0 1 2)
    #python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --sig-mod --e-lo 0.5 --e-bins 108 --add-gc --gc-flag $flag           > log_elo5_sig_mod_add_gc_gc_flag$flag.log
    #python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --bkg-mod --e-lo 0.5 --e-bins 108 --add-gc --gc-flag $flag           > log_elo5_bkg_mod_add_gc_gc_flag$flag.log
    #python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --sig-mod --bkg-mod --e-lo 0.5 --e-bins 108 --add-gc --gc-flag $flag > log_elo5_bkg_and_sig_mod_add_gc_gc_flag$flag.log
    #python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --e-lo 0.5 --e-bins 108 --add-gc --gc-flag $flag                     > log_elo5_no_mod_add_gc_gc_flag$flag.log
end

#exit


################################################################################
# Do the bulk of all the fits.
################################################################################

@ count = 1

set bins = (108 104 100 96 92)

set tags = (5 6 7 8 9)

#foreach ecut( 0.5 )
foreach ecut( 0.5 0.6 0.7 0.8 0.9 )

    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --sig-mod            --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_sig_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --bkg-mod            --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_bkg_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --bkg-mod --sig-mod  --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_bkg_and_sig_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b                      --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_no_mod.log

    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --sig-mod            --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_sig_mod_add_gc_flag3.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --bkg-mod            --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_bkg_mod_add_gc_flag3.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --bkg-mod --sig-mod  --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_bkg_and_sig_mod_add_gc_flag3.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b                      --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_no_mod_add_gc_flag3.log

    @ count += 1

end
