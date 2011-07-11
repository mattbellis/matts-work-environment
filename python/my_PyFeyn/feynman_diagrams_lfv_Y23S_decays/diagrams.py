from pyfeyn.user import *

corners = []
corners.append(Point(-3.5,1.5))
corners.append(Point(-3.5,1.5))
corners.append(Point(3.5,-1.5))
corners.append(Point(3.5,1.5))

#################################################################
def bbbar_ss_loop(outfilename, index=0):

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

    if index==4:
        Zgamma = Photon(vtx0,vtx1).addLabel(r"$Z/Z'$")
    else:
        Zgamma = Photon(vtx0,vtx1).addLabel(r"$\gamma /Z$")

    l0in = Point(2, 1)
    l1in = Point(2, -1)
    l0out = Point(3, 1)
    l1out = Point(3, -1)

    styles0 = [CYAN,THICK3]
    styles1 = [YELLOW,THICK3]

    # Set the style of the fermions based on if they are the same generation
    # or not.
    if index==0 or index==5:
        styles0 = [CYAN,THICK3]
        styles1 = [CYAN,THICK3]
    else:
        styles0 = [CYAN,THICK3]
        styles1 = [YELLOW,THICK3]

    ############################################################################
    if index==0:

        f0 = Fermion(vtx1, l0out).addLabel(r"$\ell^-$",pos=1.10,displace=0.01).setStyles(styles0)
        f1 = Fermion(vtx1, l1out).addLabel(r"$\ell^+$",pos=1.10,displace=0.01).setStyles(styles1)

        #f0 = Fermion(l0in, l0out).addLabel(r"$\ell$",pos=1.20,displace=0.01).setStyles(styles0)
        #f1 = Fermion(l1in, l1out).addLabel(r"$\ell'$",pos=1.20,displace=0.01).setStyles(styles1)
        #fc0 = Fermion(l0in, l1in).arcThru(vtx1).setStyles([CYAN,THICK3])


    elif index==1:

        f0 = Fermion(l0in, l0out).addLabel(r"$\ell^-$",pos=1.20,displace=0.01).setStyles(styles0)
        f1 = Fermion(l1in, l1out).addLabel(r"$\ell^{'+}$",pos=1.20,displace=0.01).setStyles(styles1)

        H0 = Higgs(vtx1, l0in).addLabel(r"$\bar{\ell}$",pos=0.50,displace=-0.25)
        H1 = Higgs(vtx1, l1in).addLabel(r"$\bar{\ell}'$",pos=0.50,displace=0.25)

        sf0 = Fermion(l0in, l1in).addLabel(r"$\tilde{\chi}^0$",pos=0.50,displace=-0.25)

    elif index==2:

        f0 = Fermion(l0in, l0out).addLabel(r"$\ell^-$",pos=1.20,displace=0.01).setStyles(styles0)
        f1 = Fermion(l1in, l1out).addLabel(r"$\ell^{'+}$",pos=1.20,displace=0.01).setStyles(styles1)

        ss0 = Fermion(vtx1, l0in).addLabel(r"$\tilde{\chi}^+$",pos=0.50,displace=-0.25)
        ss1 = Fermion(vtx1, l1in).addLabel(r"$\tilde{\chi}^-$",pos=0.50,displace=0.25)

        nubar = Higgs(l0in, l1in).addLabel(r"$\bar{\nu}$",pos=0.50,displace=-0.25)

    elif index==3:

        f0 = Fermion(Point(l0in.x()+0.5,l0in.y()), l0out).addLabel(r"$\ell^-$",pos=1.20,displace=0.01).setStyles(styles0)
        f1 = Fermion(Point(l1in.x()+0.5,l1in.y()), l1out).addLabel(r"$\ell^{'+}$",pos=1.20,displace=0.01).setStyles(styles1)

        h0 = Higgs(l0in, Point(l0in.x()+0.5,l0in.y())).addLabel(r"$H$",pos=0.5,displace=-0.15)
        g0 = Photon(l1in, Point(l1in.x()+0.5,l1in.y())).addLabel(r"$\gamma/Z$",pos=0.5,displace=0.15).setAmplitude(0.1)

        ss0 = Fermion(vtx1, l0in).addLabel(r"$\tilde{\chi}^+$",pos=0.50,displace=-0.25)
        ss1 = Fermion(vtx1, l1in).addLabel(r"$\tilde{\chi}^-$",pos=0.50,displace=0.25)

        top = Fermion(l0in, l1in).addLabel(r"$t$",pos=0.50,displace=+0.10)
        ell = Fermion(Point(l0in.x()+0.5,l0in.y()), Point(l1in.x()+0.5,l1in.y())).addLabel(r"$\ell^-$",pos=0.50,displace=-0.0)

    elif index==4:

        f0 = Fermion(vtx1, l0out).addLabel(r"$\ell^-$",pos=1.10,displace=0.01).setStyles(styles0)
        f1 = Fermion(vtx1, l1out).addLabel(r"$\ell^{'+}$",pos=1.10,displace=0.01).setStyles(styles1)

    elif index==5 or index==6:

        f0 = Fermion(l0in, l0out).addLabel(r"$\ell^-$",pos=1.20,displace=0.01).setStyles(styles0)
        if index==5:
            f1 = Fermion(l1in, l1out).addLabel(r"$\ell^{+}$",pos=1.20,displace=0.01).setStyles(styles1)
        else:
            f1 = Fermion(l1in, l1out).addLabel(r"$\ell^{'+}$",pos=1.20,displace=0.01).setStyles(styles1)

        nu0 = Fermion(vtx1, l0in).addLabel(r"$\nu$",pos=0.50,displace=-0.25).setStyles([CYAN,THICK3])

        if index==5:
            nu1 = Fermion(vtx1, l1in).addLabel(r"$\bar{\nu}$",pos=0.50,displace=0.25).setStyles([CYAN,THICK3])
        elif index==6:
            nu1 = Fermion(vtx1, Point(vtx1.x()+0.5,vtx1.y()-0.5)).addLabel(r"$\bar{\nu}$",pos=0.50,displace=0.25).setStyles([CYAN,THICK3])
            nu2 = Fermion(Point(vtx1.x()+0.5,vtx1.y()-0.5),l1in).addLabel(r"$\bar{\nu}'$",pos=0.50,displace=0.25).setStyles([YELLOW,THICK3])

        #nu1 = Fermion(vtx1, l1in).addLabel(r"$\bar{\nu}$",pos=0.50,displace=0.25)

        W = Higgs(l0in, l1in).addLabel(r"$W$",pos=0.50,displace=-0.25)

    ############################################################################
    # Write the output file
    ############################################################################
    fd.draw(outfilename + ".pdf")



#################################################################
def bbbar_leptoq_4pt(outfilename, index=0):

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
    q0out = Point(3, 1)
    q1out = Point(3, -1)

    vtx0 = Vertex(0.0, 1.0)
    vtx1 = Vertex(0.0, -1.0)
    vtx2 = Vertex(0.0, 0.0)

    styles0 = [CYAN,THICK3]
    styles1 = [YELLOW,THICK3]

    # Set the style of the fermions based on if they are the same generation
    # or not.
    if index==0:
        styles0 = [CYAN,THICK3]
        styles1 = [YELLOW,THICK3]
    else:
        styles0 = [CYAN,THICK3]
        styles1 = [YELLOW,THICK3]

    ############################################################################
    if index==0:

        f0 = Fermion(q0in, q0out).addLabel(r"\Pqb",pos=-0.05,displace=-0.00)
        f1 = Fermion(q1in, q1out).addLabel(r"\Paqb",pos=-0.05,displace=-0.00)

        l0 = Fermion(vtx0, q0out).addLabel(r"$\ell$",pos=1.10,displace=0.01).setStyles(styles0)
        l1 = Fermion(vtx1, q1out).addLabel(r"$\ell'$",pos=1.10,displace=0.01).setStyles(styles1)

        H0 = Higgs(vtx0, vtx1).addLabel(r"$L$",pos=0.50,displace=-0.25)


    elif index==1:

        f0 = Fermion(q0in, vtx2).addLabel(r"\Pqb",pos=-0.05,displace=-0.00)
        f1 = Fermion(q1in, vtx2).addLabel(r"\Paqb",pos=-0.05,displace=-0.00)

        l0 = Fermion(vtx2, q0out).addLabel(r"$\ell$",pos=1.10,displace=0.01).setStyles(styles0)
        l1 = Fermion(vtx2, q1out).addLabel(r"$\ell'$",pos=1.10,displace=0.01).setStyles(styles1)

        c = Circle(center=vtx2,radius=0.3,fill=[GREEN]).addLabel(r"$\alpha_{\ell \ell'} / \Lambda^2_{\ell \ell'}$",angle=90,displace=0.51)


    ############################################################################
    # Write the output file
    ############################################################################
    fd.draw(outfilename + ".pdf")



