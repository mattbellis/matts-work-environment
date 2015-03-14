from pyfeyn.user import *
from math import *

import numpy as np

corners = []
corners.append(Point(-4.0,-3.0))
corners.append(Point(-4.0,2.5))
corners.append(Point(4.5,-3.0))
corners.append(Point(4.5,2.5))

#################################################################
#################################################################

initial_fermion_names = [
        [r"$p$",r"$p$"],
        ]
fermion_names = [
        [r"$t$",r"$\bar{t}$"],
        [r"$t$",r"$\bar{t}$"],
        [r"$t$",r"$\bar{t}$"],
        [r"$t$",r"$\bar{t}$"],
        [r"$t$",r"$\bar{t}$"],
        [r"$t$",r"$\bar{t}$"],
        [r"$e^{+}$",r"$e^{-}$"],
        [r"$e^{+}$",r"$e^{-}$"],
        [r"$e^{+}$",r"$e^{-}$"],
        [r"$e^{+}$",r"$e^{-}$"],
        ]

fermion_length = [
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        ]

fragmentation_images = [4,7,10,13,16,19,20]

fragmentation_names = [
        [r"\Large{$\ell^\pm$}",r"\Large{$j$}",r"\Large{$j$}",r"q"],
        [r"\Large{$j$}",r"\Large{$j$}",r"\Large{$j_b$}",r"q"],
        ]

def pp_ttbar_BNV(outfilename, stage=0):

    processOptions()
    fd = FeynDiagram()

    ############################################################################
    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))
    ###############

    ############################################################################
    ##### Beams
    ############################################################################
    em_in = Point(-3, 0)
    em_out = Point(-0.15, 0)

    ep_in = Point(3, 0)
    ep_out = Point(0.15, 0)

    em = Fermion(em_in, em_out).addLabel(initial_fermion_names[0][0], pos=-0.05, displace=0.01).addArrow(1.05)
    ep = Fermion(ep_in, ep_out).addLabel(initial_fermion_names[0][1], pos=-0.10, displace=0.00).addArrow(1.05)
    ############################################################################

    ##### Fermions
    f_angle = 60.0
    angle = (f_angle/360.0)*2*3.14

    ############################################################################
    # Top quarks 
    ############################################################################

    f1_x = -0.05; f1_y = -0.05; f1_len = -fermion_length[0]
    f2_x = 0.05;  f2_y = 0.05;  f2_len = fermion_length[0]

    f1_x_out = f1_x+f1_len*cos(angle)
    f1_y_out = f1_y+f1_len*sin(angle)
    f1_in = Point(f1_x,f1_y)
    f1_out = Point(f1_x_out, f1_y_out)

    f2_x_out = f2_x+f2_len*cos(angle)
    f2_y_out = f2_y+f2_len*sin(angle)
    f2_in = Point(f2_x,f2_y)
    f2_out = Point(f2_x_out,f2_y_out)

    if stage>-1:
        f1 = Fermion(f1_in, f1_out).addLabel(r'$t$', pos=1.00, displace=0.10).addArrow(1.05).setStyles([CARNATIONPINK,THICK2])
        f2 = Fermion(f2_in, f2_out).addLabel(r'$\bar{t}$', pos=1.05, displace=0.18).addArrow(1.05).setStyles([CARNATIONPINK,THICK2])

    if stage>-1:
        boson1_x_out = f1_out.x()+f1_len*cos(angle+np.deg2rad(20))
        boson1_y_out = f1_out.y()+f1_len*sin(angle+np.deg2rad(20))
        boson1_in = Point(f1_out.x(),f1_out.y())
        boson1_out = Point(boson1_x_out, boson1_y_out)

        boson2_x_out = f2_out.x()+f2_len*cos(angle+np.deg2rad(20))
        boson2_y_out = f2_out.y()+f2_len*sin(angle+np.deg2rad(20))
        boson2_in = Point(f2_out.x(),f2_out.y())
        boson2_out = Point(boson2_x_out, boson2_y_out)

        f1 = Fermion(boson1_in, boson1_out).addLabel(r'$W$', pos=0.60, displace=-0.15).addArrow(1.05).setStyles([YELLOW,THICK2])
        f2 = Fermion(boson2_in, boson2_out).addLabel(r'$X$', pos=0.60, displace=-0.18).addArrow(1.05).setStyles([LIMEGREEN,THICK2])

    if stage >= -1:
        frag_angle = [240, 260, 220]
        frags = []
        for k,a in enumerate(frag_angle):
            angle = np.deg2rad(a)
            length = 1.5
            if k<2:
                xin = boson1_x_out + 0.10*cos(angle)
                yin = boson1_y_out + 0.10*sin(angle)
                xout = xin + length*cos(angle)
                yout = yin + length*sin(angle)
                fin = Point(xin, yin)
                fout = Point(xout, yout)
            else: # bjet
                xin = f1_x_out + 0.10*cos(angle)
                yin = f1_y_out + 0.10*sin(angle)
                xout = xin + (length+0.3)*cos(angle)
                yout = yin + (length+0.3)*sin(angle)
                fin = Point(xin, yin)
                fout = Point(xout, yout)
            if stage>=-1:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]))
            if stage>=-1:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]).addLabel(fragmentation_names[1][k], pos=1.09, displace=-0.10))


    if stage >=-1:
        frag_angle = [110, 80, 45]
        frags = []
        for k,a in enumerate(frag_angle):
            angle = np.deg2rad(a)
            length = 1.0
            if k<2:
                xin = boson2_x_out + 0.10*cos(angle)
                yin = boson2_y_out + 0.10*sin(angle)
                xout = xin + length*cos(angle)
                yout = yin + length*sin(angle)
                fin = Point(xin, yin)
                fout = Point(xout, yout)
            else:
                xin = f2_x_out + 0.10*cos(angle)
                yin = f2_y_out + 0.10*sin(angle)
                xout = xin + (length+0.5)*cos(angle)
                yout = yin + (length+0.5)*sin(angle)
                fin = Point(xin, yin)
                fout = Point(xout, yout)
            if stage>=-1:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]).addLabel(fragmentation_names[0][k], pos=1.09, displace=-0.10))


    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    print outfilename
    fd.draw(outfilename)


