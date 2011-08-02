from pyfeyn.user import *
from math import *

corners = []
corners.append(Point(-3.5,-1.8))
corners.append(Point(-3.5,2.5))
corners.append(Point(3.7,-0.8))
corners.append(Point(3.7,2.5))

#################################################################
#################################################################

fermion_names = [
        [r"$e^{-}$",r"$e^{+}$"],
        [r"$e^+/\mu^+$",r"$e^-/\mu^-$"],
        [r"$e^+/\mu^+$",r"$\tau^-$"],
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

def epem_lfv_y23s(outfilename, stage=0):
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

    em = Fermion(em_in, em_out).addLabel(fermion_names[0][0], pos=-0.05, displace=-0.10).addArrow(1.05)
    ep = Fermion(ep_in, ep_out).addLabel(fermion_names[0][1], pos=-0.10, displace=+0.10).addArrow(1.05)
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
    f1_len = -2.0
    #f1_len = -fermion_length[name_index]
    f1_x_out = f1_x+f1_len*cos(angle)
    f1_y_out = f1_y+f1_len*sin(angle)
    f1_in = Point(f1_x,f1_y)
    f1_out = Point(f1_x_out, f1_y_out)

    if stage==2:
        f1 = Fermion(f1_in, f1_out).addLabel(fermion_names[name_index][0], pos=0.75, displace=0.210).addArrow(1.05).setStyles([FORESTGREEN,THICK3])
    elif stage>0 and stage<3:
        f1 = Fermion(f1_in, f1_out).addLabel(fermion_names[name_index][0], pos=0.75, displace=0.210).addArrow(1.05).setStyles([CARNATIONPINK,THICK3])
    if stage>=3:
        f1 = Fermion(f1_in, f1_out).addLabel(fermion_names[name_index][0], pos=0.75, displace=0.210).addArrow(1.05).setStyles([FORESTGREEN,THICK3])


    '''
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
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([GRAY,THICK1]))
            else:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([GRAY,THICK1]).addLabel(fragmentation_names[1][k], pos=1.09, displace=-0.10))
        '''

    ############################################################################
    # Fermion 2
    ############################################################################

    angle = (f_angle/360.0)*2*3.14
    f2_x = 0.05
    f2_y = 0.05
    #f2_len = 1.5
    if stage==1:
        f2_len = 2.0
    else:
        f2_len = fermion_length[name_index]
    f2_x_out = f2_x+f2_len*cos(angle)
    f2_y_out = f2_y+f2_len*sin(angle)
    f2_in = Point(f2_x,f2_y)
    f2_out = Point(f2_x_out,f2_y_out)

    if stage==1:
        f2 = Fermion(f2_in, f2_out).addLabel(fermion_names[name_index][1], pos=0.75, displace=0.240).addArrow(1.05).setStyles([CARNATIONPINK,THICK3])
    elif stage>1:
        name_index = 2
        f2 = Fermion(f2_in, f2_out).addLabel(fermion_names[name_index][1], pos=0.75, displace=0.10).addArrow(1.05).setStyles([CARNATIONPINK,THICK3])


    if stage in fragmentation_images:
        frag_angle = [10, 40, 70]
        frags = []

        frag_names = [r"$e^-/\mu^-$", r"$\bar{\nu}_{\ell}$", r"$\nu_{\tau}$"]

        for k,a in enumerate(frag_angle):

            angle = (a/360.0)*2*3.14
            len = 1.0
            xin = f2_x_out + 0.10*cos(angle)
            yin = f2_y_out + 0.10*sin(angle)
            xout = xin + len*cos(angle)
            yout = yin + len*sin(angle)
            fin = Point(xin, yin)
            fout = Point(xout, yout)

            if stage==3 and k!=0:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([GRAY,THICK1]).addLabel(frag_names[k], pos=1.18, displace=-0.00))
            elif stage==3 and k==0:
                frags.append(Fermion(fin, fout).addArrow(1.05).setStyles([FORESTGREEN,THICK3]).addLabel(frag_names[k], pos=0.69, displace=+0.25))
            elif stage==4 and k!=0 and k!=1:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([GRAY,THICK1]).addLabel(frag_names[k], pos=1.18, displace=+0.00))
            elif stage==4 and k==0:
                frags.append(Higgs(fin, fout).addArrow(1.05).setStyles([SKYBLUE,THICK3]).addLabel(r"$\rho^-/a_1^-$", pos=0.69, displace=+0.25))

                d_frag_angle = [-10,20,60]
                d_frag_names = [r"$\pi$", r"$\pi$", r"$\pi$"]

                for kd,ad in enumerate(d_frag_angle):

                    angle = (ad/360.0)*2*3.14
                    len = 1.0
                    xin = fout.x() + 0.10*cos(angle)
                    yin = fout.y() + 0.10*sin(angle)
                    xout = xin + len*cos(angle)
                    yout = yin + len*sin(angle)
                    dfin = Point(xin, yin)
                    dfout = Point(xout, yout)

                    frags.append(Fermion(dfin, dfout).addArrow(1.05).setStyles([YELLOW,THICK3]).addLabel(d_frag_names[kd], pos=1.19, displace=+0.00))

    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    fd.draw(outfilename)
