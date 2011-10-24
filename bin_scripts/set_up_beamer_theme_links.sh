#!/bin/tcsh

set wildcard = $1

foreach file($MY_BEAMER_THEMES/*$1*)
    echo $file
    ln -s $file
end

ln -s $MY_BEAMER_THEMES/all_colors.sty .

