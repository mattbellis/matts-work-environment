from pyfeyn.user import *
from pyx import *

from subprocess import call

corners = []
corners.append(Point(-3.5,1.5))
corners.append(Point(-3.5,1.5))
corners.append(Point(3.5,-1.5))
corners.append(Point(3.5,1.5))

col = color.cmyk.Black
#text.set(mode="latex")
text.preamble(r"\usepackage{color}")
text.preamble(r"\definecolor{COL}{cmyk}{%(c)g,%(m)g,%(y)g,%(k)g}" % col.color)


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

    q0in = Point(-3, -0)
    q0out = Point(+3, -0)

    vtx0 = Vertex(-2.0, -0.5)

    q1in = Point(-3, -1)
    q1out = Point(+3, -1)

    f0 = Fermion(q0in, vtx0).addLabel(r"{\Large $\bar{b}$}",pos=-0.10,displace=+0.01).addArrow()
    f1 = Fermion(vtx0, q1in).addLabel(r"{\Large $u$}",pos=1.10,displace=-0.01).addArrow()

    # W-nu-ell
    vtx1 = Vertex(vtx0.x()+1.0,vtx0.y())
    
    W0 = Photon(vtx0,vtx1).addLabel(r"{\Large $W^+$}",pos=0.5,displace=+0.46).setAmplitude(0.1)

    styles0 = [THICK3,CYAN]
    styles0a = [THICK6,WHITE]
    styles0b = [THICK3,WHITE]
    styles1 = None
    styles1a = None
    styles1b = [THICK3,WHITE]

    f4name = None

    # Nu end
    vtx2 = Vertex(vtx1.x()+2.0,vtx1.y())
    vtx3 = Vertex(vtx2.x()+1.0,vtx2.y())


    if index==0:

        f3 = Fermion(vtx1, vtx2).setStyles(styles0a)
        f3 = Fermion(vtx1, vtx2).addLabel(r"{\Large $\nu$}",pos=0.5,displace=+0.19).setStyles(styles0)

        styles1 = [THICK3,CYAN]
        styles1b = [THICK3,WHITE]
        f4name = r"{\Large $\ell^{-}$}"

        W1 = Photon(vtx2,vtx3).addLabel(r"{\Large $W^-$}",pos=0.5,displace=+0.46).setAmplitude(0.1)

    elif index==1:

        vtx2a = vtx2.midpoint(vtx1)

        styles1 = [THICK6,CYAN]
        styles1a = [THICK3,WHITE]
        styles1b = [THICK3,CYAN]
        f4name = r"{\Large $\ell^{+}$}"

        f3 = Fermion(vtx1, vtx2a).setStyles(styles0a)
        f3 = Fermion(vtx1, vtx2a).addLabel(r"{\Large $\nu$}",pos=0.5,displace=0.29).setStyles(styles0)
        f3bar = Fermion(vtx2a, vtx2).addLabel(r"{\Large $\bar{\nu}$}",pos=0.5,displace=0.29).setStyles(styles1)
        f3bar = Fermion(vtx2a, vtx2).setStyles(styles1a)

        W1 = Photon(vtx2,vtx3).addLabel(r"{\Large $W^+$}",pos=0.5,displace=+0.46).setAmplitude(0.1)

    elif index==2 or index==3:

        vtx2a = vtx2.midpoint(vtx1)
        f4name = r"{\Large $\ell^{'+}$}"

        styles1 = [THICK6,YELLOW]
        styles1a = [THICK3,WHITE]
        styles1b = [THICK3,YELLOW]

        f3 = Fermion(vtx1, vtx2a).setStyles(styles0a)
        f3 = Fermion(vtx1, vtx2a).addLabel(r"{\Large $\nu$}",pos=0.5,displace=0.29).setStyles(styles0)
        f3bar = Fermion(vtx2a, vtx2).addLabel(r"{\Large $\bar{\nu}$}",pos=0.5,displace=0.29).setStyles(styles1)
        f3bar = Fermion(vtx2a, vtx2).setStyles(styles1a)

        W1 = Photon(vtx2,vtx3).addLabel(r"{\Large $W^+$}",pos=0.5,displace=+0.46).setAmplitude(0.1)


    f3out = Point(q0out.x()-1.0,q0out.y()+2)
    f4out = Point(q0out.x()-0.5,q0out.y()+1)

    print f4name
    f3 = Fermion(vtx1, f3out).addLabel(r"{\Large $\ell^{+}$}",pos=1.04,displace=+0.005).addArrow().setStyles(styles0)
    f4 = Fermion(f4out, vtx2).addLabel(f4name,pos=-0.11,displace=+0.005).addArrow().setStyles(styles1b)

    # Final state quarks
    q2 = Fermion(q0out, vtx3).addLabel(r"{\Large $\bar{u}$}",pos=-0.10,displace=-0.08).addArrow()
    q3 = Fermion(vtx3, q1out).addLabel(r"{\Large $d/s$}",pos=1.10,displace=+0.11).addArrow()

    if index==3:

        ecenter = q0in.midpoint(q0out)
        ecenter = Vertex(ecenter.x(),ecenter.y()-0.7)
        e = Ellipse(center=ecenter,xradius=2.5,yradius=0.7,fill=[WHITE]).addLabel(r"\textcolor{COL}{\bf New Physics}")


    ############################################################################
    # Write the output file
    ############################################################################
    fd.draw(outfilename + ".pdf")
    pdfname = "%s.pdf" % (outfilename)
    pngname = "%s.png" % (outfilename)

    call(["convert", pdfname,pngname])



