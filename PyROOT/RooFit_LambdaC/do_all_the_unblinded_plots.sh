#!/bin/tcsh 

#set tag = 'toy_embedded_signal0_correct_sys_0534'
#set tag = 'toy_embedded_signal10_correct_sys_0101'

set tag = 'unblinded_data'

foreach file(rootWorkspaceFiles/*LambdaC*ntp1*"$tag"_sig*.root)
    echo $file
    python plot_fit_results.py --workspace $file --num-bins-mes 30 --num-bins-de 20 --num-bins-nn 34 --batch 
end

foreach file(rootWorkspaceFiles/*LambdaC*ntp2*"$tag"_sig*.root)
    python plot_fit_results.py --workspace $file --num-bins-mes 30 --num-bins-de 20 --num-bins-nn 22 --batch 
end

foreach file(rootWorkspaceFiles/*Lambda0*ntp*"$tag"_sig*.root)
    python plot_fit_results.py --workspace $file --num-bins-mes 30 --num-bins-de 20 --num-bins-nn 35 --batch 
end

set tag = 'unblinded_data_fixed_PDF_to_be_positive'
foreach file(rootWorkspaceFiles/*Lambda0*ntp4*"$tag"_sig*.root)
    python plot_fit_results.py --workspace $file --num-bins-mes 30 --num-bins-de 20 --num-bins-nn 35 --batch 
end
