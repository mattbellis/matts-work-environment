#!/bin/tcsh 

foreach num (5 6 7 8 9)
    python summarize_log_files.py log_elo"$num"_*mod.log | tail -32
    python summarize_log_files.py log_elo"$num"_*add_gc_flag2.log | tail -32
    #python summarize_log_files.py log_elo"$num"_*mod.log | head -22
    #python summarize_log_files.py log_elo"$num"_*add_gc_flag2.log | head -22
end
