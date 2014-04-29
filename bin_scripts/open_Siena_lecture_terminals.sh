#!/bin/tcsh

set lecture = $1

cd
if ( -d ~/Siena/Phys_283_Modern_Physics/Fall_2011/my_lectures/lecture_$lecture ) then

    rm -f current_lecture 
    ln -s ~/NIU/Phys_283_Modern_Physics/Fall_2011/my_lectures/lecture_$lecture current_lecture 

    mrxvt -name NIU_lecture_notes &
    mrxvt -name NIU_lecture_slides &

else

    echo "Directory doesn't exist!"

endif

