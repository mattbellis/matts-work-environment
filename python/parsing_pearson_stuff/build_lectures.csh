#!/bin/tcsh

set cmd = 'python parsing_pearson_stuff.py'

set dir = "~/Work/Teaching/Pearson/Giancoli_Physics_Sixth_Edition/Chap_1-10/" 

$cmd $dir"/Chapter_01/Present/Images/" "Introduction, Measurement, Estimating" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_1_Introduction_Measurement_Estimating/content.tex
$cmd $dir"/Chapter_02/Present/Images/" "Kinematics in one dimension" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_2_Describing_Motion_Kinematics_in_One_Dimension/content.tex
$cmd $dir"/Chapter_03/Present/Images/" "Kinematics in two dimensions; Vectors" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_3_Kinematics_in_Two_Dimensions_Vectors/content.tex
$cmd $dir"/Chapter_04/Present/Images/" "Newton's Laws of Motion" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_4_Dynamics_Newtons_Laws_of_Motion/content.tex
$cmd $dir"/Chapter_05/Present/Images/" "Circular motion; Gravitation" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_5_Circular_Motion_Gravitation/content.tex

$cmd $dir"/Chapter_06/Present/Images/" "Work and energy" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_6_Work_and_Energy/content.tex
echo $cmd $dir"/Chapter_07/Present/Images/" "Linear momentum" 
$cmd $dir"/Chapter_07/Present/Images/" "Linear momentum" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_7_Linear_Momentum/content.tex
$cmd $dir"/Chapter_08/Present/Images/" "Rotational motion" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_8_Rotational_Motion/content.tex
$cmd $dir"/Chapter_09/Present/Images/" "Static equilibrium; Elasticity and fracture" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_9_Static_Equilibrium_Elasticity_and_Fracture/content.tex
$cmd $dir"/Chapter_10/Present/Images/" "Fluids" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_10_Fluids/content.tex

################################################################################
set dir = "~/Work/Teaching/Pearson/Giancoli_Physics_Sixth_Edition/Chap_11-22/" 

$cmd $dir"/Chapter_11/Present/Images/" "Vibrations and waves" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_11_Vibrations_and_Waves/content.tex
$cmd $dir"/Chapter_12/Present/Images/" "Sound" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_12_Sound/content.tex
$cmd $dir"/Chapter_13/Present/Images/" "Temperature and kinetic theory" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_13_Temperature_and_Kinetic_Theory/content.tex
$cmd $dir"/Chapter_14/Present/Images/" "Heat" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_14_Heat/content.tex
$cmd $dir"/Chapter_15/Present/Images/" "The laws of thermodynamics" > ~/Siena/courses/PHYS_110_General_Physics_IA/lecture_slides/Lecture_15_The_Laws_of_Thermodynamics/content.tex
