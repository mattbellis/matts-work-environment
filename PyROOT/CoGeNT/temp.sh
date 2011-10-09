#!/bin/tcsh 

foreach num (5 6 7 8 9)
    python summarize_log_files.py log_elo"$num"_*mod.log | tail -24
    python summarize_log_files.py log_elo"$num"_*flag2.log | tail -24
end
