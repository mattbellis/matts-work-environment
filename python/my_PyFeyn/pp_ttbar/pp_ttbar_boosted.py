from pyfeyn.user import *
from math import *

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
        1.0,
        1.0,
        ]

fragmentation_images = [4,7,10,13,16,19,20]

fragmentation_names = [
        [r"$e^-$",r"$\bar{\nu}_e$",r"$D^+$",r"q"],
        [r"$\mu^+$",r"$\nu_\mu$",r"$D^-$",r"q"],
        ]

def pp_ttbar_boosted(outfilename, stage=0):

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

    if stage >= -1:
        frag_angle = [185, 210, 245]
        frags = []
        for k,a in enumerate(frag_angle):
            angle = (a/360.0)*2*3.14
            len = 1.5
            xin = f1_x_out + 0.10*cos(angle)
            yin = f1_y_out + 0.10*sin(angle)
            xout = xin + len*cos(angle)
            yout = yin + len*sin(angle)
            fin = Point(xin, yin)
            fout = Point(xout, yout)
            if stage>=-1:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]))
            if stage>=-1:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]).addLabel(fragmentation_names[1][k], pos=1.09, displace=-0.10))


    if stage >=-1:
        frag_angle = [60, 20, 350]
        frags = []
        for k,a in enumerate(frag_angle):
            angle = (a/360.0)*2*3.14
            len = 1.0
            xin = f2_x_out + 0.10*cos(angle)
            yin = f2_y_out + 0.10*sin(angle)
            xout = xin + len*cos(angle)
            yout = yin + len*sin(angle)
            fin = Point(xin, yin)
            fout = Point(xout, yout)
            if stage>=-1:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]).addLabel(fragmentation_names[0][k], pos=1.09, displace=-0.10))


    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    fd.draw(outfilename)


