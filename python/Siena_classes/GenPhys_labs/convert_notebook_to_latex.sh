/opt/anaconda3/bin/jupyter nbconvert --to latex $1
# Then edit the .tex file to comment out the
#    \maketitle
# and add a 
#    \small
# after it. 
#
# Also manually change
# matplotlib inline
#
# to 
#
# matplotlib notebook
