foreach file ($*)

    python fit_data.py $file | grep nsig
    
end
