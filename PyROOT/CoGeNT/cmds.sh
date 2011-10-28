#!/bin/tcsh 


################################################################################
# Do some studies to look at the different errors for the Gaussian constraint
################################################################################
#foreach flag(0 1 2)
foreach flag(2)
    #python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --exp-mod --e-lo 0.5 --e-bins 108 --add-gc --gc-flag $flag           > log_elo5_exp_mod_add_gc_gc_flag$flag.log
    #python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --flat-mod --e-lo 0.5 --e-bins 108 --add-gc --gc-flag $flag           > log_elo5_flat_mod_add_gc_gc_flag$flag.log
    #python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --exp-mod --flat-mod --e-lo 0.5 --e-bins 108 --add-gc --gc-flag $flag > log_elo5_flat_and_exp_mod_add_gc_gc_flag$flag.log
    #python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --cg-mod --exp-mod --flat-mod --e-lo 0.5 --e-bins 108 --add-gc --gc-flag $flag > log_elo5_flat_and_exp_and_mod_add_gc_gc_flag$flag.log
    #python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --e-lo 0.5 --e-bins 108 --add-gc --gc-flag $flag                     > log_elo5_no_mod_add_gc_gc_flag$flag.log
end

#exit


################################################################################
# Do the bulk of all the fits.
################################################################################

@ count = 1

set bins = (108 104 100 96 92)

set tags = (5 6 7 8 9)

#foreach ecut( 0.6 )
foreach ecut( 0.5 0.6 0.7 0.8 0.9 )

    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --exp-mod            --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_exp_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --flat-mod            --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_flat_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --cg-mod            --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_cg_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --flat-mod --exp-mod  --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_flat_and_exp_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --cg-mod --flat-mod --exp-mod  --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_flat_and_exp_and_cg_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b                      --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags[$count]"_no_mod.log

    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --exp-mod            --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_exp_mod_add_gc_flag2.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --flat-mod            --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_flat_mod_add_gc_flag2.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --cg-mod            --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_cg_mod_add_gc_flag2.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --flat-mod --exp-mod  --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_flat_and_exp_mod_add_gc_flag2.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --cg-mod --flat-mod --exp-mod  --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_flat_and_exp_and_cg_mod_add_gc_flag2.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b                      --e-lo $ecut --e-bins $bins[$count] --gc-flag 2 --add-gc > log_elo"$tags[$count]"_no_mod_add_gc_flag2.log

    @ count += 1

end
