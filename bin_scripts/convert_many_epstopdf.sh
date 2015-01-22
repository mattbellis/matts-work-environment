#!/bin/tcsh

foreach file($*)

    echo $file
    epstopdf $file

end
