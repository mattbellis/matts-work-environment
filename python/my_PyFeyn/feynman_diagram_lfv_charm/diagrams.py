from pyfeyn.user import *

corners = []
corners.append(Point(-3.5,2.5))
corners.append(Point(-3.5,2.5))
corners.append(Point(3.5,-2.5))
corners.append(Point(3.5,2.5))

#################################################################
def fd_lfv_c_quark(outfilename, index=0):

    processOptions()
    fd = FeynDiagram()

    ############################################################################
    # Make border
    ############################################################################
    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))

    q0in = Point(-3, -1)
    q0out = Point(3, -1)

    vtx0 = Vertex(-1.0, -1)
    vtx1 = Vertex(+1.0, -1)

    f0 = Fermion(q0in, vtx0).addLabel(r"\Pqc",pos=-0.10,displace=+0.01).addArrow()
    f1 = Fermion(vtx1, q0out).addLabel(r"\Pqu",pos=1.10,displace=-0.01).addArrow()

    f2 = Fermion(vtx0, vtx1).addArrow()

    if index==0 or index==1:

        vtx2 = Vertex(vtx0.x(),vtx0.y()+1.0)
        vtx3 = Vertex(vtx1.x(),vtx1.y()+1.0)

        W0 = Photon(vtx0,vtx2).addLabel(r"$W$",pos=0.5,displace=-0.25).setAmplitude(0.1)
        W1 = Photon(vtx1,vtx3).addLabel(r"$W$",pos=0.5,displace=-0.25).setAmplitude(0.1)

        styles0 = [THICK3,CYAN]
        styles1 = None
        f4name = None
        if index==0:
            styles1 = [THICK3,CYAN]
            f4name = r"$\ell^{+}$"
        elif index==1:
            styles1 = [THICK3,YELLOW]
            f4name = r"$\ell^{'+}$"


        if index==0:

            f3 = Fermion(vtx2, vtx3).addLabel(r"$\nu$",pos=0.5,displace=-0.10).setStyles(styles0)

        elif index==1:

            vtx5 = vtx3.midpoint(vtx2)

            f3 = Fermion(vtx2, vtx5).addLabel(r"$\nu$",pos=0.5,displace=0.10).setStyles(styles0)
            f3bar = Fermion(vtx5, vtx3).addLabel(r"$\bar{\nu}$",pos=0.5,displace=0.10).setStyles(styles1)


        f3out = Point(q0out.x(),q0out.y()+3)
        f4out = Point(q0out.x(),q0out.y()+2)

        f3 = Fermion(vtx2, f3out).addLabel(r"$\ell^{-}$",pos=1.04,displace=+0.005).addArrow().setStyles(styles0)
        f4 = Fermion(f4out, vtx3).addLabel(f4name,pos=-0.11,displace=+0.005).addArrow().setStyles(styles1)

    elif index==2 or index==3:

        vtx2 = vtx0.midpoint(vtx1)
        vtx3 = Vertex(vtx1.x(),vtx0.y()+2.0)

        W0 = Photon(vtx0,vtx1).addLabel(r"$W$",pos=0.5,displace=+0.25).setAmplitude(0.1).bend(1.0)

        zgamma = Photon(vtx2,vtx3).addLabel(r"$\gamma /Z$",pos=0.5,displace=-0.35).setAmplitude(0.2)

        f3out = Point(q0out.x(),q0out.y()+3)
        f4out = Point(q0out.x(),q0out.y()+1)

        styles0 = [THICK3,CYAN]
        styles1 = None
        f4name = None
        if index==2:
            styles1 = [THICK3,CYAN]
            f4name = r"$\ell^{+}$"
        elif index==3:
            styles1 = [THICK3,YELLOW]
            f4name = r"$\ell^{'+}$"

        f3 = Fermion(vtx3, f3out).addLabel(r"$\ell^{-}$",pos=1.08,displace=+0.005).addArrow().setStyles(styles0)
        f4 = Fermion(f4out, vtx3).addLabel(f4name,pos=-0.11,displace=+0.005).addArrow().setStyles(styles1)

    ############################################################################
    # Write the output file
    ############################################################################
    fd.draw(outfilename + ".pdf")



