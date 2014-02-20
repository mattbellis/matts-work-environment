from pyfeyn.user import *
from math import *
from numpy import *

corners = []
corners.append(Point(-4.0,-2.0))
corners.append(Point(-4.0,2.5))
corners.append(Point(4.5,-2.0))
corners.append(Point(4.5,2.5))

#################################################################
# branch a line off of something
#################################################################
def decay_line(line_pts, angle, length):

    # Convert degrees to radians
    org_x_len = line_pts[1].getX() - line_pts[0].getX()
    org_y_len = line_pts[1].getY() - line_pts[0].getY()

    org_angle = arctan2(org_y_len,org_x_len)

    print org_angle
    print rad2deg(org_angle)

    angle = deg2rad(angle)
    print angle
    angle += org_angle

    print angle

    xin = line_pts[1].getX() + 0.01
    yin = line_pts[1].getY() + 0.01

    print "angles: %f %f " % (cos(angle),sin(angle))
    print "xin/yin: %f %f " % (xin,yin)
    print "angle: %f %f " % (angle,rad2deg(angle))

    xout = xin + length*cos(angle)
    yout = yin + length*sin(angle)

    print "xout/yout: %f %f " % (xout,yout)

    fin = Point(xin, yin)
    fout = Point(xout, yout)

    return [fin,fout]

#################################################################
#################################################################

def D_mix_final_state_simple_version(outfilename, stage=0):
    #from pyfeyn.user import *

    processOptions()
    fd = FeynDiagram()

    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))
    ###############

    ##### Quark lines
    q_y_pos = 0.0
    qbar_y_pos = 0.0 

    q_in = Point( -3.0, q_y_pos)
    q_out = Point( -1.0, q_y_pos)

    qbar_in = Point(1.0, qbar_y_pos)
    qbar_out = Point(3.0, qbar_y_pos)

    q = Fermion(q_in, q_out)
    q.addArrow(0.50)
    q.addLabel(r"\PDzero",pos=-0.15,displace=0.0)

    qbar = Fermion(qbar_in, qbar_out)
    qbar.addArrow(0.50)
    qbar.addLabel(r"\PaDz",pos=1.15,displace=0.00)

    f_loop_0 = Fermion(q_out,qbar_in).bend(0.75)
    f_loop_0.setStyles([color.rgb.red,THICK2])
    f_loop_0.addArrow(0.55)
    f_loop_0.addLabel(r"$K^+,\pi^+,$...",pos=0.5,displace=0.20)

    f_loop_1 = Fermion(q_out,qbar_in).bend(-0.75)
    f_loop_1.setStyles([color.rgb.blue,THICK2])
    f_loop_1.addArrow(0.55)
    f_loop_1.addLabel(r"$K^-,\pi^-,$...",pos=0.5,displace=-0.20)


    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    fd.draw(outfilename)








'''
def D_mix_finalstate(outfilename, stage):

    ##### D's
    Dangle = 30.0
    angle = (Dangle/360.0)*2*3.14

    # Fragmenting B
    Dbarx = -0.05
    Dbary = -0.05
    Dbar_len = -2.0
    Dbarx_out = Dbarx+Dbar_len*cos(angle)
    Dbary_out = Dbary+Dbar_len*sin(angle)
    Dbar_in = Point(Dbarx,Dbary)
    Dbar_out = Point(Dbarx_out, Dbary_out)

    if stage>0:
        Dbar = Fermion(Dbar_in, Dbar_out).addLabel(r"\APDzero", pos=1.00, displace=0.18).addArrow(1.05)

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
    D_len = 1.5
    Dx_out = Dx+D_len*cos(angle)
    Dy_out = Dy+D_len*sin(angle)
    D_in = Point(Dx,Dy)
    D_out = Point(Dx_out,Dy_out)

    if stage>0:
        D = Fermion(D_in, D_out).addLabel(r"\PDzero", pos=1.05, displace=0.18).addArrow(1.05).setStyles([color.rgb.red,THICK3])

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
        muon = Fermion(fin, fout).addArrow(1.05).addLabel(r"\Plm",pos=0.9,displace=0.20).setStyles([color.rgb.blue,THICK2])

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
        lam = Fermion(fin, fout).addArrow(1.05).addLabel(r"\PcgLp",pos=0.9,displace=0.20).setStyles([color.rgb.red,THICK2])

    # proton
    lam_decay_angles = [60.0,25.0,-15.0]
    lam_decay_strings = ["p","Km","piplus"]
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
'''
