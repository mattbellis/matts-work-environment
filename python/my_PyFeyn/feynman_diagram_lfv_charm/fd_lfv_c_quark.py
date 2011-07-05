from pyfeyn.user import *

corners = []
corners.append(Point(-4.5,-3.0))
corners.append(Point(-4.5,2.0))
corners.append(Point(4.5,-3.0))
corners.append(Point(4.5,2.0))

#################################################################
def fd_lfv_c_quark(outfilename, mesonname, q_m0, q_m1, q_b1, q_b2, lep, baryonname, doEllipse=0):

    print "Here: " + mesonname + " " + baryonname

    processOptions()
    fd = FeynDiagram()

    define_border = []
    for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))

    if doEllipse:
        reference0 = Ellipse(x=3.3,y=0.2,xradius=0.2,yradius=1.5,stroke=[pyx.color.rgb.white],fill=[pyx.color.rgb.black])
        reference0.setDepth(0.0)

    in1 = Point(-3, 1)
    in2 = Point(-3, 0)
    out1 = Point(3, 1)
    out2 = Point(3, 0)

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

    fd.draw(outfilename + ".pdf")



