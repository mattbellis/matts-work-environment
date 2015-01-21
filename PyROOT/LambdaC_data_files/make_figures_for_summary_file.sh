#!/bin/tcsh 

#set file_tag = 'toy_embedded_signal0_correct_sys_0534'
#set file_tag = 'toy_embedded_signal10_correct_sys_0101'
set file_tag = 'unblinded_data_sig'

set sizewhich = "width"
set sizenums = ( "0.99" "0.99" "0.45" "0.45" )

set baryons = ( "LambdaC" "LambdaC" "Lambda0" "Lambda0" "Lambda0" "Lambda0" )
set ntps = ( "ntp1" "ntp2" "ntp1" "ntp2" "ntp3" "ntp4" )
set tags = ( "dim3_nfits" "dim3_nfits" "dim2_nfits" "dim2_nfits" "dim2_nfits" "dim2_nfits" )
set tags_noGC = ( "dim3_noGC_nfits" "dim3_noGC_nfits" "dim2_noGC_nfits" "dim2_noGC_nfits" "dim2_noGC_nfits" "dim2_noGC_nfits" )

set caption_reactions = ('B^0\\rightarrow\Lambda_c^+ \mu^-' \
                        'B^0\\rightarrow\Lambda_c^+ e^-' \
                        'B^-\\rightarrow\Lambda^0 \mu^-' \
                        'B^-\\rightarrow\Lambda^0 e^-' \
                        'B^-\\rightarrow\\bar{\Lambda}^0 \mu^-' \
                        'B^-\\rightarrow\\bar{\Lambda}^0 e^-' \
                        )


set tex_file = '~/BaBar/BNV_unblinded_results/sections/unblinded.tex'
rm $tex_file >& /dev/null

############# Make the tables ################

echo '\\begin{table}' >> $tex_file
echo '\\caption{Extracted branching fractions, upper-limits and signficance for the decay modes of interest.}' >> $tex_file
echo "\\begin{tabular}{l || c || c | c | c || c}" >> $tex_file
echo '\hline' >> $tex_file
echo 'Reaction & UL (90\%) ($\\times 10^{-8}$) & BF ($\\times 10^{-8}$) & BF tot. err. & BF stat. err. & Significance ($\sigma$) \\\\ ' >> $tex_file
echo '\hline' >> $tex_file
echo '\hline' >> $tex_file

@ i = 1
while ( $i < 7 )

    set baryon = $baryons[$i]
    set ntp = $ntps[$i]
    set tag = $tags[$i]
    set tag_noGC = $tags_noGC[$i]

    if ( $i == 3 ) then
        echo '\hline' >> $tex_file
    endif

    set bf = `sed -n 1p fit_summary_log_files/*$baryon*$ntp**$file_tag*$tag*.log | awk '{print 100*$2}'`
    set tot_err = `sed -n 1p fit_summary_log_files/*$baryon*$ntp**$file_tag*$tag*.log | awk '{print 100*$4}'`
    set stat_err = `sed -n 1p fit_summary_log_files/*$baryon*$ntp**$file_tag*$tag_noGC*.log | awk '{print 100*$4}'`

    set tot_err_lo = `sed -n 1p fit_summary_log_files/*$baryon*$ntp**$file_tag*$tag*.log | awk '{printf "%3.3f", 100*$7}'`
    set tot_err_hi = `sed -n 1p fit_summary_log_files/*$baryon*$ntp**$file_tag*$tag*.log | awk '{printf "%3.3f", 100*$8}'`
    set stat_err_lo = `sed -n 1p fit_summary_log_files/*$baryon*$ntp**$file_tag*$tag_noGC*.log | awk '{printf "%3.3f", 100*$7}'`
    set stat_err_hi = `sed -n 1p fit_summary_log_files/*$baryon*$ntp**$file_tag*$tag_noGC*.log | awk '{printf "%3.3f", 100*$8}'`

    #set tot_err_lo = `printf "%3.3f" $tot_err_lo`
    #echo $tot_err_lo

    #echo $tot_err " " $stat_err
    set sys_err = `echo $tot_err $stat_err | awk '{print sqrt($1*$1 - $2*$2)'}`

    set ul = `sed -n 2p fit_summary_log_files/*$baryon*$ntp**$file_tag*$tag*.log | awk '{print $2}'`
    set sigma = `sed -n 3p fit_summary_log_files/*$baryon*$ntp**$file_tag*$tag*.log | awk '{print $2}'`

    #echo '$'" $caption_reactions[$i] "'$' " & $ul & $bf " '$\pm$'  "$stat_err (stat)" '$\pm$'  "$sys_err (sys) &  $sigma \\\\" >> $tex_file
    echo '$'" $caption_reactions[$i] "'$' " & $ul & $bf  &  $tot_err_lo,$tot_err_hi & $stat_err_lo,$stat_err_hi &  $sigma \\\\" >> $tex_file

    @ i ++

end

echo '\hline' >> $tex_file
echo '\\end{tabular}' >> $tex_file
echo '\\end{table}' >> $tex_file
echo '\\newpage' >> $tex_file
echo '\\clearpage' >> $tex_file
echo  >> $tex_file
echo  >> $tex_file
echo  >> $tex_file



############# Make the plots ################

@ count = 0

@ i = 1
while ( $i < 7 )

    set baryon = $baryons[$i]
    set ntp = $ntps[$i]
    set tag = $tags[$i]

    foreach plot ( 0 1 2 3 )

        @ j = $plot + 1
        set sizenum = $sizenums[$j]

        foreach file(Plots/*$baryon*$ntp**$file_tag*$tag*_"$plot".eps)

            epstopdf $file
            set newfile = `basename $file eps`pdf

            cp Plots/$newfile ~/BaBar/BNV_unblinded_results/figures/
            echo $newfile


            if ( $plot != "3" ) then
                echo >> $tex_file
                echo "\\begin{figure}[H]" >> $tex_file
            endif

            echo "\\includegraphics[$sizewhich=$sizenum\\text$sizewhich]{figures/$newfile}" >> $tex_file

            if ( $plot != "2" ) then
                echo '\\caption{$'$caption_reactions[$i]'$}' >> $tex_file
                echo "\\label{lab$count}" >> $tex_file
                echo "\\end{figure}" >> $tex_file
                echo >> $tex_file
            endif

            @ count += 1

        end
    end

    @ i++
end


foreach file(Plots/*$file_tag*.eps)
    epstopdf $file
end
cp Plots/*$file_tag*.pdf ~/BaBar/PRD_RC_BtoLambdaLepton//figures
cp Plots/*$file_tag*.pdf ~//BaBar/myBADs/BNV_Bdecays_BAD//figures
