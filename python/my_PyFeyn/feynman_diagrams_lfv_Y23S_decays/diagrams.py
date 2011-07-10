from pyfeyn.user import *

corners = []
corners.append(Point(-3.5,1.5))
corners.append(Point(-3.5,1.5))
corners.append(Point(3.5,-1.5))
corners.append(Point(3.5,1.5))

#################################################################
def bbbar_ss_loop(outfilename, iloop=0):

    processOptions()
    fd = FeynDiagram()

    ############################################################################
    # Make border
    ############################################################################
    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))

    q0in = Point(-3, 1)
    q1in = Point(-3, -1)
    q0out = Point(-2, 1)
    q1out = Point(-2, -1)

    vtx0 = Vertex(-1.0, q0in.midpoint(q1in).y())

    f0 = Fermion(q0in, q0out).addLabel(r"\Pqb",pos=-0.10,displace=-0.00)
    fc0 = Fermion(q0out, q1out).arcThru(vtx0)
    f1 = Fermion(q1in, q1out).addLabel(r"\Paqb",pos=-0.10,displace=-0.00)

    vtx1 = Vertex(1.0, q0in.midpoint(q1in).y())

    Zgamma = Photon(vtx0,vtx1).addLabel(r"\Pgg /\PZ")

    l0in = Point(2, 1)
    l1in = Point(2, -1)
    l0out = Point(3, 1)
    l1out = Point(3, -1)

    f0 = Fermion(l0in, l0out).addLabel(r"$\ell$",pos=1.20,displace=0.01)
    f1 = Fermion(l1in, l1out).addLabel(r"$\ell'$",pos=1.20,displace=0.01)

    H0 = Higgs(vtx1, l0in).addLabel(r"$\bar{\ell}$",pos=0.50,displace=-0.25)
    H1 = Higgs(vtx1, l1in).addLabel(r"$\bar{\ell}'$",pos=0.50,displace=0.25)

    sf0 = Fermion(l0in, l1in).addLabel(r"$\tilde{\chi}^0$",pos=0.50,displace=-0.25)

    '''
    out_vtx = Vertex(0, 0, mark=CIRCLE)

    xdecay_vtx = Vertex(1.5, -1.5, mark=CIRCLE)

    c1 = Circle(center=out_vtx, radius=0.2, fill=[pyx.color.cmyk.Yellow], points = [out_vtx])
    c2 = Circle(center=xdecay_vtx, radius=0.2, fill=[color.rgb.green], points = [xdecay_vtx])

    decay1 = Point(3,-1)
    decay2 = Point(3,-2)

    l1 = Label("\P"+mesonname, x=-4.0, y=0.5)
    l2 = Label("\P"+baryonname, x=4, y=0.2)

    fa1 = Fermion(in1, out1).addLabel(r"\P"+q_m0, pos=-0.05, displace=0.01).addLabel(r"\P"+q_m0, pos=1.05, displace=0.00).setStyles([WHITE])
    fa2 = Fermion(in2, out2).addLabel(r"\P"+q_m1, pos=-0.05, displace=0.01).addLabel(r"\P"+q_b1, pos=1.05, displace=0.00).setStyles([WHITE])
    fx  = Photon(out_vtx, xdecay_vtx).addLabel(r"X").setStyles([WHITE])
    fx_decay1  = Fermion(xdecay_vtx, decay1).addLabel(r"\P"+q_b2, pos=1.23, displace=0.00).setStyles([WHITE])
    fx_decay2  = Fermion(xdecay_vtx, decay2).addLabel(r"\P"+lep, pos=1.28, displace=0.02).setStyles([WHITE])
    '''

    fd.draw(outfilename + ".pdf")



