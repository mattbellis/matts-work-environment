from pyfeyn.user import *
from math import *

corners = []
corners.append(Point(-4.0,-3.0))
corners.append(Point(-4.0,2.5))
corners.append(Point(4.5,-3.0))
corners.append(Point(4.5,2.5))

#################################################################
#################################################################

def epem_ccbar_D0_to_Kshh(outfilename, stage=0):
    #from pyfeyn.user import *

    processOptions()
    fd = FeynDiagram()

    ####### Set common borders
    #corners = []
    #corners.append(Point(-4.5,-3.5))
    #corners.append(Point(-4.5,3.5))
    #corners.append(Point(4.5,-3.5))
    #corners.append(Point(4.5,3.5))

    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))
    ###############

    ##### Beams
    em_in = Point(-3, 0)
    em_out = Point(-0.15, 0)

    ep_in = Point(3, 0)
    ep_out = Point(0.15, 0)

    em = Fermion(em_in, em_out).addLabel(r"\Pem", pos=-0.05, displace=0.01).addArrow(1.05)
    ep = Fermion(ep_in, ep_out).addLabel(r"\Pep", pos=-0.10, displace=0.00).addArrow(1.05)

    ##### D's
    Dangle = 30.0
    angle = (Dangle/360.0)*2*3.14

    # Fragmenting B
    Dbarx = -0.05
    Dbary = -0.05
    Dbar_len = -1.0
    Dbarx_out = Dbarx+Dbar_len*cos(angle)
    Dbary_out = Dbary+Dbar_len*sin(angle)
    Dbar_in = Point(Dbarx,Dbary)
    Dbar_out = Point(Dbarx_out, Dbary_out)

    if stage>0:
        Dbar = Fermion(Dbar_in, Dbar_out).addLabel(r"\Paqc", pos=0.80, displace=-0.18).addArrow(1.05)

    frag_angle = [185, 200, 215, 240]
    frags = []
    for a in frag_angle:
        angle = (a/360.0)*2*3.14
        len = 1.5
        xin = Dbarx_out + 0.10*cos(angle)
        yin = Dbary_out + 0.10*sin(angle)
        xout = xin + len*cos(angle)
        yout = yin + len*sin(angle)
        fin = Point(xin, yin)
        fout = Point(xout, yout)
        if stage>1:
            frags.append(Higgs(fin, fout).addArrow(1.05))

    # Signal B
    angle = (Dangle/360.0)*2*3.14
    Dx = 0.05
    Dy = 0.05
    D_len = 1.0
    Dx_out = Dx+D_len*cos(angle)
    Dy_out = Dy+D_len*sin(angle)
    D_in = Point(Dx,Dy)
    D_out = Point(Dx_out,Dy_out)

    if stage>0:
        D = Fermion(D_in, D_out).addLabel(r"D$^{*+}$", pos=0.35, displace=-0.12).addArrow(1.05).setStyles([color.rgb.red,THICK3])

    # muon
    angle = (70.0/360.0)*2*3.14
    len = 1.0
    xin = Dx_out + 0.10*cos(angle)
    yin = Dy_out + 0.10*sin(angle)
    xout = xin + len*cos(angle)
    yout = yin + len*sin(angle)
    fin = Point(xin, yin)
    fout = Point(xout, yout)
    if stage>2:
        muon = Fermion(fin, fout).addArrow(1.05).addLabel(r"\Ppiplus",pos=0.9,displace=-0.20).setStyles([color.rgb.blue,THICK2])

    # Lambda_c
    angle = (20.0/360.0)*2*3.14
    len = 1.0
    xin = Dx_out + 0.10*cos(angle)
    yin = Dy_out + 0.10*sin(angle)
    lam_xout = xin + len*cos(angle)
    lam_yout = yin + len*sin(angle)
    fin = Point(xin, yin)
    fout = Point(lam_xout, lam_yout)
    if stage>2:
        lam = Fermion(fin, fout).addArrow(1.05).addLabel(r"\PDzero",pos=0.75,displace=0.20).setStyles([color.rgb.red,THICK2])

    # proton
    lam_decay_angles = [60.0,25.0,-10.0]
    lam_decay_strings = ["K$^0_S$","h$^{+}$","h$^{-}$"]
    lam_decay_strings = ["x$_0$","x$_1$","x$_2$"]
    lam_decay = []
    for i,a in enumerate(lam_decay_angles):
        angle = (a/360.0)*2*3.14
        len = 1.0
        xin = lam_xout + 0.10*cos(angle)
        yin = lam_yout + 0.10*sin(angle)
        xout = xin + len*cos(angle)
        yout = yin + len*sin(angle)
        fin = Point(xin, yin)
        fout = Point(xout, yout)
        if stage>3:
            lam_decay.append(Fermion(fin, fout).addArrow(1.05).addLabel(r""+lam_decay_strings[i],pos=1.3,displace=0.00).setStyles([color.rgb.blue,THICK2]))




    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    fd.draw(outfilename)
