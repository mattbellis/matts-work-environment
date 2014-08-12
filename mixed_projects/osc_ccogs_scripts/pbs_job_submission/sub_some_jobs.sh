#python build_submission_files.py d100k_f100k_000 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index000.dat
#python build_submission_files.py d100k_f100k_001 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index001.dat
#python build_submission_files.py d100k_f100k_002 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index002.dat
#python build_submission_files.py d100k_f100k_003 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index003.dat
#python build_submission_files.py d100k_f100k_004 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index004.dat
#python build_submission_files.py d100k_f100k_005 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index005.dat
#python build_submission_files.py d100k_f100k_006 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index006.dat
#python build_submission_files.py d100k_f100k_007 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index007.dat
#python build_submission_files.py d100k_f100k_008 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index008.dat
#python build_submission_files.py d100k_f100k_009 weschler_z250-275_100k_arcseconds.dat flat_100k_arcseconds_max10000_index009.dat

#python build_submission_files.py d1M_f10M_000 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index000.dat
#python build_submission_files.py d1M_f10M_001 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index001.dat
#python build_submission_files.py d1M_f10M_002 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index002.dat
#python build_submission_files.py d1M_f10M_003 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index003.dat
#python build_submission_files.py d1M_f10M_004 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index004.dat
#python build_submission_files.py d1M_f10M_005 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index005.dat
#python build_submission_files.py d1M_f10M_006 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index006.dat
#python build_submission_files.py d1M_f10M_007 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index007.dat
#python build_submission_files.py d1M_f10M_008 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index008.dat
#python build_submission_files.py d1M_f10M_009 weschler_z250-275_1M_arcseconds.dat flat_10M_arcseconds_max1000000_index009.dat

foreach file0(flat_10M_arcseconds_max1000000_index000.dat flat_10M_arcseconds_max1000000_index001.dat flat_10M_arcseconds_max1000000_index002.dat flat_10M_arcseconds_max1000000_index003.dat flat_10M_arcseconds_max1000000_index004.dat flat_10M_arcseconds_max1000000_index005.dat flat_10M_arcseconds_max1000000_index006.dat flat_10M_arcseconds_max1000000_index007.dat flat_10M_arcseconds_max1000000_index008.dat flat_10M_arcseconds_max1000000_index009.dat)
    foreach file1(flat_10M_arcseconds_max1000000_index000.dat flat_10M_arcseconds_max1000000_index001.dat flat_10M_arcseconds_max1000000_index002.dat flat_10M_arcseconds_max1000000_index003.dat flat_10M_arcseconds_max1000000_index004.dat flat_10M_arcseconds_max1000000_index005.dat flat_10M_arcseconds_max1000000_index006.dat flat_10M_arcseconds_max1000000_index007.dat flat_10M_arcseconds_max1000000_index008.dat flat_10M_arcseconds_max1000000_index009.dat)

    set i0 = `basename $file0 .dat | awk -F"index" '{print $2}'`
    set i1 = `basename $file1 .dat | awk -F"index" '{print $2}'`
    set tag = "d1M_f10M_ff_"$i0"_"$i1

    #echo $i0
    #echo $i1

    #@ i0test = `echo $i0 | awk '{print $1}'`
    #echo $i0test
    #@ i1test = `echo $i1 | awk '{print $1}'`
    #echo $i1test

    #if ( $i1test >= $i0test ) then

        echo python build_submission_files.py $tag $file0 $file1 flatflat
             python build_submission_files.py $tag $file0 $file1 flatflat
     #endif

    end
end
