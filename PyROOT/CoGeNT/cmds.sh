#!/bin/tcsh 


@ count = 1

set bins = (120, 110, 100, 90, 80)

set tags = (5, 6, 7, 8, 9)

foreach ecut( 0.5 0.6 0.7 0.8 0.9 )

    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --sig-mod           --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags"_sig_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --bkg-mod           --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags"_bkg_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --bkg-mod --sig-mod --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags"_bkg_and_sig_mod.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b                     --e-lo $ecut --e-bins $bins[$count]   > log_elo"$tags"_no_mod.log

    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --sig-mod           --e-lo $ecut --e-bins $bins[$count]   --add-gc > log_elo"$tags"_sig_mod_add_gc.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --bkg-mod           --e-lo $ecut --e-bins $bins[$count]   --add-gc > log_elo"$tags"_bkg_mod_add_gc.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b --bkg-mod --sig-mod --e-lo $ecut --e-bins $bins[$count]   --add-gc > log_elo"$tags"_bkg_and_sig_mod_add_gc.log
    python2.7 read_in_from_text_file.py data/before_fire_LG.dat -b                     --e-lo $ecut --e-bins $bins[$count]   --add-gc > log_elo"$tags"_no_mod_add_gc.log

    @ count += 1

end
