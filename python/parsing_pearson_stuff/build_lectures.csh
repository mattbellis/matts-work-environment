#!/bin/tcsh

set cmd = 'python parsing_pearson_stuff.py'

set dir = "~/Work/Teaching/Pearson/Giancoli_Physics_Sixth_Edition/Chap_1-10/" 

$cmd $dir"/Chapter_01/Present/Images/" "Introduction, Measurement, Estimating" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_1_Introduction_Measurement_Estimating/content.tex
$cmd $dir"/Chapter_02/Present/Images/" "Kinematics in one dimension" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_2_Describing_Motion_Kinematics_in_One_Dimension/content.tex
