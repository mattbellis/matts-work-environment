touch log_toys.log
foreach file($*)
    echo $file
    python fit_data.py $file | grep nsig >> log_toys.log
end
