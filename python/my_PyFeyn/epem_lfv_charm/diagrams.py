from pyfeyn.user import *
from math import *

corners = []
corners.append(Point(-4.0,-3.0))
corners.append(Point(-4.0,2.5))
corners.append(Point(4.5,-3.0))
corners.append(Point(4.5,2.5))

#################################################################
#################################################################

fermion_names = [
        [r"$e^{-}$",r"$e^{+}$"],
        [r"$c$",r"$\bar{c}$"],
        [r"$q$",r"$\bar{q}$"],
        ]

fermion_length = [
        1.0,
        1.0,
        ]

fragmentation_images = [3,4,5]

fragmentation_names = [
        [r"$\pi^+$",r"$\pi^-$",r"$\mu^-$",r"$\bar{\nu_{\mu}}$"],
        [r"$\pi^+$",r"$\pi^-$",r"$K^0$",r"$K^-$"]
        ]

def epem_lfv_charm(outfilename, stage=0):
    #from pyfeyn.user import *

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

    em = Fermion(em_in, em_out).addLabel(fermion_names[0][0], pos=-0.05, displace=0.01).addArrow(1.05)
    ep = Fermion(ep_in, ep_out).addLabel(fermion_names[0][1], pos=-0.10, displace=0.00).addArrow(1.05)
    ############################################################################

    ##### Fermions
    f_angle = 30.0
    angle = (f_angle/360.0)*2*3.14

    name_index = 1

    ############################################################################
    # Fermion 1
    ############################################################################

    f1_x = -0.05
    f1_y = -0.05
    f1_len = -fermion_length[name_index]
    f1_x_out = f1_x+f1_len*cos(angle)
    f1_y_out = f1_y+f1_len*sin(angle)
    f1_in = Point(f1_x,f1_y)
    f1_out = Point(f1_x_out, f1_y_out)

    if stage>0:
        f1 = Fermion(f1_in, f1_out).addLabel(fermion_names[name_index][0], pos=1.00, displace=0.18).addArrow(1.05).setStyles([CARNATIONPINK,THICK3])

    if stage>1:
        f4 = Fermion(Point(f1_x+0.1,f1_y-0.15), Point(f1_x_out+0.1,f1_y_out-0.15)).addLabel(fermion_names[-1][1], pos=1.05, displace=-0.18).addArrow(1.05).setStyles([APRICOT,THICK1])

    if stage in fragmentation_images:
        frag_angle = [185, 200, 215, 240]
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
            if stage<=19:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]))
            else:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]).addLabel(fragmentation_names[1][k], pos=1.09, displace=-0.10))
            '''
            if stage>1:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]))
            '''

    ############################################################################
    # Fermion 2
    ############################################################################

    angle = (f_angle/360.0)*2*3.14
    f2_x = 0.05
    f2_y = 0.05
    #f2_len = 1.5
    f2_len = fermion_length[name_index]
    f2_x_out = f2_x+f2_len*cos(angle)
    f2_y_out = f2_y+f2_len*sin(angle)
    f2_in = Point(f2_x,f2_y)
    f2_out = Point(f2_x_out,f2_y_out)

    if stage>0:
        f2 = Fermion(f2_in, f2_out).addLabel(fermion_names[name_index][1], pos=1.05, displace=0.18).addArrow(1.05).setStyles([CARNATIONPINK,THICK3])

    if stage>1:
        f4 = Fermion(Point(f2_x-0.1,f2_y+0.15), Point(f2_x_out-0.1,f2_y_out+0.15)).addLabel(fermion_names[-1][0], pos=1.05, displace=-0.18).addArrow(1.05).setStyles([APRICOT,THICK1])

    if stage in fragmentation_images:
        frag_angle = [350, 20, 60, 80]
        frags = []

        for k,a in enumerate(frag_angle):

            angle = (a/360.0)*2*3.14
            len = 1.5
            xin = f2_x_out + 0.10*cos(angle)
            yin = f2_y_out + 0.10*sin(angle)
            xout = xin + len*cos(angle)
            yout = yin + len*sin(angle)
            fin = Point(xin, yin)
            fout = Point(xout, yout)

            if stage<4 or (stage==4 and k!=1):
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([CYAN,THICK2]))
            else:
                frags.append(Fermion(fin, fout).addArrow(1.05).setStyles([CYAN,THICK3]).addLabel(r"$D/D_{s}/\Lambda_c$", pos=0.69, displace=+0.20))

                d_frag_angle = [-10,30,80]
                d_frag_names = [r"$h$", r"$\ell$", r"$\ell$"]

                for kd,ad in enumerate(d_frag_angle):

                    angle = (ad/360.0)*2*3.14
                    len = 1.0
                    xin = fout.x() + 0.10*cos(angle)
                    yin = fout.y() + 0.10*sin(angle)
                    xout = xin + len*cos(angle)
                    yout = yin + len*sin(angle)
                    dfin = Point(xin, yin)
                    dfout = Point(xout, yout)

                    if kd==0:
                        frags.append(Higgs(dfin, dfout).addArrow(1.05).setStyles([CYAN,THICK2]).addLabel(d_frag_names[kd], pos=1.09, displace=+0.10))
                    else:
                        frags.append(Higgs(dfin, dfout).addArrow(1.05).setStyles([YELLOW,THICK2]).addLabel(d_frag_names[kd], pos=1.09, displace=+0.10))

    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    fd.draw(outfilename)
