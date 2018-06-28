set dir = `pwd`

echo $dir 

foreach file($*)

    ls $dir/$file

    set newfile = `basename $file ipynb`py
    pwd
    jupyter-nbconvert --to python $dir/$file

    mv $newfile $dir/.

    #cd $currentdir
end
