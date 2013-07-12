from pyfeyn.user import *
from math import *

corners = []
corners.append(Point(-4.0,-2.0))
corners.append(Point(-4.0,2.0))
corners.append(Point(4.0,-2.0))
corners.append(Point(4.0,2.0))

#################################################################
#################################################################

def lfv_loop_slepton(outfilename, stage=0):
    #from pyfeyn.user import *

    processOptions()
    fd = FeynDiagram()

    ####### Set common borders
    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))
    ###############

    ##### Initial state fermions
    f0_i = Point(-3.5, 1.5)
    f0_f = Point(-2, 0)

    f1_i = Point(-3.5, -1.5)
    f1_f = Point(-2, 0)

    f0 = Fermion(f0_i, f0_f).addLabel("$b_R$", pos=-0.05, displace=0.01).addArrow(0.50)
    f1 = Fermion(f1_i, f1_f).addLabel("$s_L$", pos=-0.10, displace=0.00).addArrow(0.50)

    ##### Final state fermions
    f2_i = Point(2, 0)
    f2_f = Point(3.5, 1.5)

    f3_i = Point(2, 0)
    f3_f = Point(3.5, -1.5)

    f2 = Fermion(f2_i, f2_f).addLabel("$\mu_R$", pos=1.10, displace=0.01).addArrow(0.50)
    f3 = Fermion(f3_i, f3_f).addLabel("$\\bar{\\nu}_\\tau$", pos=1.10, displace=0.00).addArrow(0.50)

    f2.setStyles([color.rgb.red,THICK2])
    f3.setStyles([color.rgb.blue,THICK2])

    loops = Circle(center=f2_i, radius=0.2, fill=[YELLOW], points=[f2_i])


    # Propagator
    higgs = Higgs(f1_f, f2_i).addLabel("$H^+$", pos=0.50, displace=-0.10)

    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    fd.draw(outfilename)


########################################################################3
def b_leptonic_decay(outfilename, stage=0):
    #from pyfeyn.user import *

    processOptions()
    fd = FeynDiagram()

    ####### Set common borders
    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))
    ###############

    ##### Initial state fermions
    f0_i = Point(-3.5, 1.5)
    f0_f = Point(-2, 0)

    f1_i = Point(-3.5, -1.5)
    f1_f = Point(-2, 0)

    f0 = Fermion(f0_f, f0_i).addLabel("$\\bar{b}$", pos=-0.05, displace=0.01).addArrow(0.50)
    f1 = Fermion(f1_i, f1_f).addLabel("$u$", pos=-0.10, displace=0.00).addArrow(0.50)

    B_blob = Ellipse(x=-3.5, y=0, xradius=1, yradius=1.5).setFillStyle(CROSSHATCHED45)

    ##### Final state fermions
    f2_i = Point(2, 0)
    f2_f = Point(3.5, 1.5)

    f3_i = Point(2, 0)
    f3_f = Point(3.5, -1.5)

    f2 = Fermion(f2_i, f2_f).addLabel("$\mu_R$", pos=1.10, displace=0.01).addArrow(0.50)
    f3 = Fermion(f3_i, f3_f).addLabel("$\\bar{\\nu}_\\tau$", pos=1.10, displace=0.00).addArrow(0.50)

    f2.setStyles([color.rgb.red,THICK2])
    f3.setStyles([color.rgb.blue,THICK2])

    loops = Circle(center=f2_i, radius=0.2, fill=[YELLOW], points=[f2_i])


    # Propagator
    higgs = Higgs(f1_f, f2_i).addLabel("$H^+$", pos=0.50, displace=-0.10)

    # Print the file
    outfilename = "%s_%d.pdf" % (outfilename,stage)
    fd.draw(outfilename)
