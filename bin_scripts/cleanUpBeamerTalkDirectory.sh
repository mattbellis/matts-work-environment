#!/bin/tcsh -f

#mkdir -p temp_figures
#mkdir -p temp_roofit_figures
#mkdir -p temp_tmva_figures

#grep figures talk.tex | grep "\.pdf"

foreach dir ( "tmva_figures" "roofit_figures" "figures" )
    set new_dir = "temp_"$dir

    mkdir $new_dir


    grep includegraphics talk.tex | grep "{$dir\/" | awk -F"{" '{print $2}'  | awk -F"}" '{print $1}'
    echo "-----------------"
    foreach file(`grep includegraphics talk.tex | grep "{$dir\/" | awk -F"includegraphics" '{print $2}' | awk -F"{" '{print $2}'  | awk -F"}" '{print $1}'`)

        set new_file = `echo $file | sed s/$dir/$new_dir/`
        echo $file
        echo $new_file
        echo cp $file $new_file
        cp $file $new_file

    end

mv $dir $dir.BAK
mv $new_dir $dir
#mv tempFigures figures

end
