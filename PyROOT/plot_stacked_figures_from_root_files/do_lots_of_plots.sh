#!/bin/tcsh

set baryon = $1
set ntp = $2

#foreach baryon("LambdaC" "Lambda0")
    #foreach ntp( 1 2 3 4 )
        foreach plot(0 1 2 5 )
            foreach cut(0 1 2 3 4 5 6 7)

            set tag = "_"$baryon"ntp"$ntp"_effects_of_cuts_blind"
            echo ./plot_some_stacked_compared_to_data.py rootFiles/"$baryon"_ntp"$ntp"_*generic[BQ]*cuts_blind* rootFiles/"$baryon"_ntp"$ntp"_*sideband*cuts_blind* --hname hmass0_"$plot"_"$cut" --tag $tag --batch
            ./plot_some_stacked_compared_to_data.py rootFiles/"$baryon"_ntp"$ntp"_*generic[BQ]*cuts_blind* rootFiles/"$baryon"_ntp"$ntp"_*sideband*cuts_blind* --hname hmass0_"$plot"_"$cut" --tag $tag --batch

            end
        end
    #end
#end
