#!/bin/tcsh -f

foreach dir ( "figures" "tmva_figures" "roofit_figures" )

        if ( -e $dir ) then

            set new_dir = "temp_"$dir

            mkdir $new_dir

        foreach tex_file (*.tex)
            grep includegraphics $tex_file | grep "{$dir\/" | awk -F"{" '{print $2}'  | awk -F"}" '{print $1}'
            echo "-----------------"
            foreach file(`grep includegraphics $tex_file | grep "{$dir\/" | awk -F"includegraphics" '{print $2}' | awk -F"{" '{print $2}'  | awk -F"}" '{print $1}'`)

                set new_file = `echo $file | sed s/$dir/$new_dir/`
                #echo "Figure: " $file
                #echo "New Figure: " $new_file
                #echo cp $file $new_file
                cp $file $new_file

            end
            touch $tex_file
        end

        mv $dir $dir.BAK
        mv $new_dir $dir
        rm -rf $dir.BAK
      
       endif

end
