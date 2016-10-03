foreach file($*)

    echo $file

    rename "s/ //g" $file

end
