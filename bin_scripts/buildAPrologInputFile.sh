#!/bin/tcsh

echo "% sstate(Ja,La,Lf,Pa,P1,P2,S1,S2,R,Alpha);"
echo "lam_f(-1/2)."
echo "lam_f(1/2)."
echo
echo "lam_i(1/2)."
echo "lam_i(-1/2)."
echo
echo "lam_g(1)."
echo "lam_g(-1)."
echo
echo "%     ( Ja,   La, Lf, Pa, P1, P2,  S1,S2, R,             Alpha);"
echo "%     (  J,    M, Lf,  P, P1, P2,  S1,S2, R,             Alpha);"
echo
echo "% J: J of resonance"
echo "% M: M of resonance"
echo "% Lf: final proton helicity"
echo "%P Parity of Resonance"
echo "%P1: parity of decay 1 of resonance"
echo "%P2: parity of decay 2 of resonance"
echo "%S1 spin of decay 1"
echo "%S2 spin of decay 2"
echo
echo "accInt('accInt','accint')."
echo "rawInt('rawInt','rawint')."
echo

foreach file($*)

#	echo $file
	
		set J = `echo $file | awk -F "" '{print $1}'`
		set P = `echo $file | awk -F "" '{print $2}'`
		set isobar = `echo $file | awk -F "\." '{print $2}'`
		set mg = `echo $file | awk -F "\." '{print $3}' | awk -F "=" '{print $2}'`
		set mi = `echo $file | awk -F "\." '{print $4}' | awk -F "=" '{print $2}'`
		set li = `echo $file | awk -F "\." '{print $5}' | awk -F "=" '{print $2}'`
		set si = `echo $file | awk -F "\." '{print $6}' | awk -F "-" '{print $1}' | awk -F "=" '{print $2}'`
		set mf = `echo $file | awk -F "\." '{print $6}' | awk -F "=" '{print $3}'`
		set lf = `echo $file | awk -F "\." '{print $7}' | awk -F "=" '{print $2}'`
		set sf = `echo $file | awk -F "\." '{print $8}' | awk -F "=" '{print $2}'`

		## Set the M of intermediate resonance based on gamma m and initial proton m
		if ($mg == "+1" && $mi == "+1") then 
			set M = "+3"
		else if ($mg == "+1" && $mi == "-1") then 
			set M = "+1"
		else if ($mg == "-1" && $mi == "+1") then 
			set M = "-1"
		else if ($mg == "-1" && $mi == "-1") then 
			set M = "-3"
		else 
			echo "Bad initial mg and mi.........."
			echo "Can only be +/- 1 for each!!! Oh nooooooooooooo!!!!"
			exit(-1)
		endif

		## Set the parities and spins based on isobar
		if ($isobar == "delta++" || $isobar == "delta0") then 
			set p1 = "+1"
			set p2 = "-1"
			set s1 = "3/2"
			set s2 = "0"
		else if ($isobar == "rho") then 
			set p1 = "-1"
			set p2 = "+1"
			set s1 = "1"
			set s2 = "1/2"
		else 
			set p1 = "+1"
			set p2 = "-1"
			set s1 = "3/2"
			set s2 = "0"
		endif

		#echo $J" "$P" "$isobar" "$mg" "$mi" "$li" "$si" "$mf" "$lf" "$sf" "$M
		if( !(($mg == "+1" && $mi == "+1" && $si == "1") || ($mg == "-1" && $mi == "-1" && $si == "1")) )  then
#			echo $file
#			vamp < $file > temp
			#head -1 temp
#			head -1 temp | getArgOfComplex 
			#if($li != "0" && $lf != "0") then
				echo "sstate($J/2,$M/2,$mf/2,$P""1,$p1,$p2,$s1,$s2,'$file','$J$P.$isobar.li=$li.si=$si.lf=$lf.sf=$sf')."
			#endif
		endif

#3-.delta++.mg=-1.mi=+1.li=2.si=1--mf=-1.lf=2.sf=3.amps

end


