foreach file ($*)

    set filename = `basename $file`
    set xsec_mass = `echo $filename | sed s/log_xsec// | sed s/_/e-/ | sed s/_/\ / | sed s/.log//`

    set lh = `grep 'final lh' $file | awk '{print $3}'`

    echo $xsec_mass $lh

end
