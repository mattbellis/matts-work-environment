#!/bin/bash 

# In the folder containing a LaTeX source beamer file named 'foo.tex'
# one types 'makehandouts foo' to create 'foo-handout.pdf' for viewing
# slides without pauses, and create also 'foo-handoutnup.pdf' where n
# = 1,2,4 for easy formatted slides printing with acroread. 
#
# Notes: Run this command in the same folder containing 'foo.tex'
#       Type 'makehandouts' to get a hint on usage


if [ -n "$1" ]; then
 echo "# makefile for $1.tex (created by makehandouts command)" > /tmp/makefile
 echo "# creates $1-handout.pdf (for viewing slides without" >> /tmp/makefile
 echo "# pauses), $1-handout1up.pdf, $1-handout2up.pdf," >> /tmp/makefile
 echo "# and $1-handout4up.pdf (formatted for printing)" >> /tmp/makefile
 echo "default:	$1-handout4up.pdf	$1-handout2up.pdf	$1-handout1up.pdf" >> /tmp/makefile
 echo " " >> /tmp/makefile
 echo "1up:	$1-handout1up.pdf" >> /tmp/makefile
 echo " " >> /tmp/makefile
 echo "2up:	$1-handout2up.pdf" >> /tmp/makefile
 echo " " >> /tmp/makefile
 echo "4up:	$1-handout4up.pdf" >> /tmp/makefile
 echo " " >> /tmp/makefile
 echo "$1-handout4up.pdf: $1-handout.pdf" >> /tmp/makefile
 echo "	pdfnup $1-handout.pdf --nup 2x2 --outfile /tmp/junk.pdf" >> /tmp/makefile
 echo "	pdf90 /tmp/junk.pdf --outfile $1-handout4up.pdf" >> /tmp/makefile
 echo " " >> /tmp/makefile
 echo "$1-handout2up.pdf:	$1-handout.pdf" >> /tmp/makefile
 echo "	pdfnup $1-handout.pdf --nup 1x2 --outfile $1-handout2up.pdf" >> /tmp/makefile
 echo " " >> /tmp/makefile
 echo "$1-handout1up.pdf:	$1-handout.pdf" >> /tmp/makefile
 echo "	pdf90 $1-handout.pdf --outfile $1-handout1up.pdf" >> /tmp/makefile
 echo " " >> /tmp/makefile
 echo "$1-handout.pdf:	$1-handout.tex" >> /tmp/makefile
 #echo "	pdflatex $1-handout" >> /tmp/makefile
 #echo "	pdflatex $1-handout" >> /tmp/makefile
 echo "	xelatex $1-handout" >> /tmp/makefile
 echo "	xelatex $1-handout" >> /tmp/makefile
 echo " " >> /tmp/makefile
 echo "$1-handout.tex:	$1.tex" >> /tmp/makefile
 echo "	sed -e 's/\\documentclass\[/\documentclass[handout,/g' -e 's/\\documentclass{/\documentclass[handout]{/g' $1.tex > $1-handout.tex" >> /tmp/makefile
 echo "created makefile for $1.tex"
else
 echo "usage: 'makehandouts <LaTeX root filename>'"
 echo "e.g.:  'makehandouts foo'  to create handouts for foo.tex"
 exit 1
fi

if [ -e $1.tex ]; then 
 echo "checking that $1.tex exists...yes"
 make -f /tmp/makefile
else
 echo "Aborting: File $1.tex does not exist! Typo?"
 exit 1
fi
