from pyfeyn.user import *

corners = []
corners.append(Point(-3.5,2.5))
corners.append(Point(-3.5,2.5))
corners.append(Point(3.5,-2.5))
corners.append(Point(3.5,2.5))

#################################################################
def neutron_decay(outfilename, index=0):

    processOptions()
    fd = FeynDiagram()

    ############################################################################
    # Make border
    ############################################################################
    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))

    q0in = Point(-3, 0)

    vtx0 = Vertex(0.0, 0)

    q0out = Point(2.7, +2)
    q1out = Point(2.7, -1.0)
    q2out = Point(2.7, -2)

    f0 = Fermion(q0in, vtx0).addLabel(r"$n$",pos=-0.10,displace=+0.01).addArrow()

    f1 = Fermion(vtx0, q0out).addLabel(r"$p$",pos=1.10,displace=-0.01).addArrow()
    f2 = Fermion(vtx0, q1out).addLabel(r"$e^-$",pos=1.10,displace=-0.01).addArrow()
    f3 = Higgs(vtx0, q2out).addLabel(r"$\bar{\nu}_e$",pos=1.10,displace=-0.01).addArrow()

    ############################################################################
    # Write the output file
    ############################################################################
    fd.draw(outfilename + ".pdf")



