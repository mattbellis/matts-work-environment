set tag = $1
foreach num( 1 2 3 4 5 6 7 8 9 )
    tcsh run_some_DM_scans.sh $num >& logrunning_"$num"_"$tag".log &
end
