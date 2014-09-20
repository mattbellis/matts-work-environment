#!/bin/tcsh 


#./plotAVariable.py hshape 7 rootFiles/hshape_SP9446_shapeStudyCuts1.root rootFiles/hshape_SP1005_shapeStudyCuts1.root
#./plotAVariable.py hmass 7 rootFiles/hmass_SP9446_shapeStudyCuts1.root rootFiles/hmass_SP1005_shapeStudyCuts1.root
#./plotAVariable.py hshape 7 rootFiles/hshape_SP9446_shapeStudyCuts1.root rootFiles/hshape_AllSP_shapeStudyCuts1.root rootFiles/hshape_OnPeak_shapeStudyCuts1.root


#foreach background( SP1005 SP998 SP1235 SP1237 SP9446 OnPeak OffPeak )

  #./plotAVariable.py h2D "$background"_shapeStudyCuts1_narrow 7 rootFiles/h2D_"$background"_shapeStudyCuts1.root  batch

#end
#./plotAVariable.py hshape sig_on_shapeStudyCuts1 7 rootFiles/hshape_SP9446_shapeStudyCuts1.root rootFiles/hshape_OnPeak_shapeStudyCuts1.root
#./plotAVariable.py hshape sig_all_shapeStudyCuts1 7 rootFiles/hshape_SP9446_shapeStudyCuts1.root rootFiles/hshape_AllSP_shapeStudyCuts1.root


./plotFromOnlyOneFile.py hmass SP9446_showSidebandSigmaCuts 10 rootFiles/hmass_SP9446_showSidebandSigmaCuts.root
./plotFromOnlyOneFile.py h2D SP9446_showSidebandSigmaCuts 10 rootFiles/h2D_SP9446_showSidebandSigmaCuts.root
./plotFromOnlyOneFile.py hmass OnPeak_showSidebandSigmaCuts 10 rootFiles/hmass_OnPeak_showSidebandSigmaCuts.root
./plotFromOnlyOneFile.py h2D OnPeak_showSidebandSigmaCuts 10 rootFiles/h2D_OnPeak_showSidebandSigmaCuts.root


./plotFromOnlyOneFile.py hmass SP9445_showSidebandSigmaCuts 2 rootFiles/hmass_SP9445_showSidebandSigmaCuts.root
