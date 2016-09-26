foreach file($*) 

    echo " -----------------------------------"
    echo " ------------ $file ----------------"
    echo " -----------------------------------"
    echo
    echo 
    ~/anaconda2/bin/python $file
    echo 
    echo
end
