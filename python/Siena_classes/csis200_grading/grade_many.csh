foreach file($*) 

    echo " -----------------------------------"
    echo " ------------ $file ----------------"
    echo " -----------------------------------"
    echo
    echo 
    ~/anaconda/bin/python $file
    echo 
    echo
end
