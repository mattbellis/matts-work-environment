from pyfeyn.user import *
from math import *

corners = []
corners.append(Point(-4.0,-3.0))
corners.append(Point(-4.0,2.5))
corners.append(Point(4.5,-3.0))
corners.append(Point(4.5,2.5))

#################################################################
#################################################################

def epem_B_lambda0(outfilename, stage=0):
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

    ##### B's
    Bangle = 30.0
    angle = (Bangle/360.0)*2*3.14

    # Fragmenting B
    Bbarx = -0.05
    Bbary = -0.05
    Bbar_len = -2.0
    Bbarx_out = Bbarx+Bbar_len*cos(angle)
    Bbary_out = Bbary+Bbar_len*sin(angle)
    Bbar_in = Point(Bbarx,Bbary)
    Bbar_out = Point(Bbarx_out, Bbary_out)

    if stage>0:
        Bbar = Fermion(Bbar_in, Bbar_out).addLabel(r"\PBm", pos=1.00, displace=0.18).addArrow(1.05)

    frag_angle = [185, 200, 215, 240]
    frags = []
    for a in frag_angle:
        angle = (a/360.0)*2*3.14
        len = 1.5
        xin = Bbarx_out + 0.10*cos(angle)
        yin = Bbary_out + 0.10*sin(angle)
        xout = xin + len*cos(angle)
        yout = yin + len*sin(angle)
        fin = Point(xin, yin)
        fout = Point(xout, yout)
        if stage>1:
            frags.append(Higgs(fin, fout).addArrow(1.05))

    # Signal B
    angle = (Bangle/360.0)*2*3.14
    Bx = 0.05
    By = 0.05
    B_len = 1.5
    Bx_out = Bx+B_len*cos(angle)
    By_out = By+B_len*sin(angle)
    B_in = Point(Bx,By)
    B_out = Point(Bx_out,By_out)

    if stage>0:
        B = Fermion(B_in, B_out).addLabel(r"\PBp", pos=1.05, displace=0.18).addArrow(1.05).setStyles([color.rgb.red,THICK3])

    # muon
    angle = (70.0/360.0)*2*3.14
    len = 1.0
    xin = Bx_out + 0.10*cos(angle)
    yin = By_out + 0.10*sin(angle)
    xout = xin + len*cos(angle)
    yout = yin + len*sin(angle)
    fin = Point(xin, yin)
    fout = Point(xout, yout)
    if stage>2:
        muon = Fermion(fin, fout).addArrow(1.05).addLabel(r"\Plp",pos=0.9,displace=0.20).setStyles([color.rgb.blue,THICK2])

    # Lambda_c
    angle = (20.0/360.0)*2*3.14
    len = 1.0
    xin = Bx_out + 0.10*cos(angle)
    yin = By_out + 0.10*sin(angle)
    lam_xout = xin + len*cos(angle)
    lam_yout = yin + len*sin(angle)
    fin = Point(xin, yin)
    fout = Point(lam_xout, lam_yout)
    if stage>2:
        lam = Fermion(fin, fout).addArrow(1.05).addLabel(r"\PgL",pos=0.9,displace=0.20).setStyles([color.rgb.red,THICK2])

    # proton
    lam_decay_angles = [40.0,0.0]
    lam_decay_strings = ["p","piminus"]
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
            lam_decay.append(Fermion(fin, fout).addArrow(1.05).addLabel(r"\P"+lam_decay_strings[i],pos=1.2,displace=0.00).setStyles([color.rgb.blue,THICK2]))




    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    fd.draw(outfilename)
