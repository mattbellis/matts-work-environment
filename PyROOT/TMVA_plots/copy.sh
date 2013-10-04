#!/bin/tcsh 

scp "bellis@yakut07.slac.stanford.edu:LeptBc_bellis/TMVA-v4.0.3/TMVA/my_analysis/rootFiles/**$1*.root" rootFiles/.
scp "bellis@yakut07.slac.stanford.edu:LeptBc_bellis/TMVA-v4.0.3/TMVA/my_analysis/logs/**$1*.log" logFiles/.

#scp "yakut02.slac.stanford.edu:~/LeptBc_bellis/TMVA/my_analysis/*$1*.log" .
