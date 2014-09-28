#!/usr/bin/env tcsh 

set fit = "2"
#set fitname = "SHM_fixed_sigma_n_free_DM"
#set fitname = "SHM_fixed_sigma_n_free_DM"
#set fitname = "SHM_scan_sigma_n_and_DM"
#set fitname = "SHM_scan_sigma_n_and_DM_fixed_neutrons_contribution_880"
set fitname = "SHM_scan_sigma_n_and_DM_free_neutrons_contribution"
#set fitname = "stream_scan_sigma_n_and_DM_fixed_neutrons_contribution_880"

set logdir = "logs_"$fitname

if ( ! -e $logdir ) then
    mkdir $logdir
endif

#ls $logdir

#set base = $1
set coef = $1

#foreach base ( 39 40 41 42 )
foreach base ( 40 41 42 )
#foreach base ( 40 )
#foreach base ( 41 )
    #foreach base ( 42 )
    #foreach coef ( 1 2 3 4 5 6 7 8 9 )
    foreach mDM ( 5 6 8 10 15 20 30 )
        #foreach mDM ( 5 )

        set sigma_n_tag = `printf "xsec%d_%d" $coef $base`
        set mDM_tag = `printf "mDM%d" $mDM`
        set val = `printf "%se-%s" $coef $base`

        set logfile = "$logdir/log_"$sigma_n_tag"_"$mDM.log

        echo python fit_cogent_data.py --fit $fit --batch --mDM $mDM --sigma_n $val --tag $fitname"_"$sigma_n_tag"_"$mDM_tag $logfile
        python fit_cogent_data.py --fit $fit --batch --mDM $mDM --sigma_n $val --tag $fitname"_"$sigma_n_tag"_"$mDM_tag >& $logfile

        # For testing purposes
        #echo python fit_cogent_data.py --fit $fit --batch --sigma_n $val --tag $fitname"_"$sigma_n_tag $logfile
        #python fit_cogent_data.py --batch --tag 'testing' >& 'logtesting'$sigma_n_tag'.log'

        end
    end
    #end
