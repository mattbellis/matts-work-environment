ls

set pdffile = `ls *.pdf`

echo $pdffile

evince $pdffile &

set workdir = `pwd`

cd ~/anaconda/bin; jupyter-notebook --notebook-dir $workdir

