#!/bin/tcsh

set today = `date +%m%d%y`
set logfile = 'arxiv_'$today'.log'

################################################################################
# Download the latest stuff from the arxiv 
################################################################################
if ( ! -e $logfile ) then

    curl -B http://arxiv.org/list/hep-ex/new >& $logfile
    curl -B http://arxiv.org/list/hep-ph/new >>& $logfile
    curl -B http://arxiv.org/list/hep-th/new >>& $logfile
    curl -B http://arxiv.org/list/nucl-ex/new >>& $logfile
    curl -B http://arxiv.org/list/gr-qc/new >>& $logfile
    curl -B http://arxiv.org/list/q-fin/new >>& $logfile
    curl -B http://arxiv.org/list/stat/new >>& $logfile

endif

################################################################################
# Search them.
################################################################################
foreach word ('baryon' 'leptoq' 'lepto-q' 'su(5' 'so(10' 'B-L' 'B+L' 'charmon' 'hybrid' 'exotic' 'penta')

    echo '------------------------------------------------------------------------------'
    echo '------------------------------------------------------------------------------'
    echo '------------------------------------------------------------------------------'
    echo '------------------------------------------------------------------------------'
    echo $logfile
    echo ' --------------------------------------- '
    echo '------------------------------------------------------------------------------'
    echo $word
    echo ' --------------------------------------- '
    grep -C 10 -i $word $logfile
    echo '------------------------------------------------------------------------------'
    echo '------------------------------------------------------------------------------'

end
