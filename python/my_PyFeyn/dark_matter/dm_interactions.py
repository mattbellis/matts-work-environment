import numpy as np
from pyfeyn.user import *


#################################################################
def dm_diagrams(outfilename, index):

    corners = []
    corners.append(Point(-4.2,-4.2))
    corners.append(Point(-4.2,4.2))
    corners.append(Point(4.2,-4.2))
    corners.append(Point(4.2,4.2))

    processOptions()
    fd = FeynDiagram()

    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))
        #define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

    in1 = Point(-3, 3)
    in2 = Point(-3,-3)
    out1 = Point(3, 3)
    out2 = Point(3,-3)

    center = Point(0,0)

    out_vtx = Vertex(-1, 1, mark=CIRCLE)

    #xdecay_vtx = Vertex(1.0, -1.0, mark=CIRCLE)

    #c1 = Circle(center=out_vtx, radius=0.1, fill=[pyx.color.cmyk.Yellow], points = [out_vtx])
    #c2 = Circle(center=xdecay_vtx, radius=0.1, fill=[color.cmyk.Green], points = [xdecay_vtx])

    decay1 = Point(3,-1)
    decay2 = Point(3,-2)

    jet_len = 2.2

    if index==0:
        a = Arrow(pos=0.3,size=0.8)
        dm1  = Fermion(in1,center).addLabel(r"\huge \bf X", pos=-0.1, displace=0.00).setStyles([APRICOT,THICK6]).addArrow(0.5,a)
        dm2  = Fermion(in2,center).addLabel(r"\huge \bf X", pos=-0.1, displace=0.00).setStyles([APRICOT,THICK6]).addArrow(0.5,a)

        a = Arrow(pos=0.9,size=0.8)
        sm1  = Fermion(center,out1).addLabel(r"\huge \bf SM", pos=1.1, displace=0.00).setStyles([CYAN,THICK6]).addArrow(0.5,a)
        sm2  = Fermion(center,out2).addLabel(r"\huge \bf SM", pos=1.2, displace=0.00).setStyles([CYAN,THICK6]).addArrow(0.5,a)

    elif index==1:
        a = Arrow(pos=0.3,size=0.8)
        dm1  = Fermion(in1,center).addLabel(r"\huge \bf X", pos=-0.1, displace=0.00).setStyles([APRICOT,THICK6]).addArrow(0.5,a)
        a = Arrow(pos=0.9,size=0.8)
        dm2  = Fermion(center,out1).addLabel(r"\huge \bf X", pos=1.1, displace=0.00).setStyles([APRICOT,THICK6]).addArrow(0.5,a)

        a = Arrow(pos=0.3,size=0.8)
        sm1  = Fermion(in2,center).addLabel(r"\huge \bf SM", pos=-0.15, displace=0.00).setStyles([CYAN,THICK6]).addArrow(0.5,a)
        a = Arrow(pos=0.9,size=0.8)
        sm2  = Fermion(center,out2).addLabel(r"\huge \bf SM", pos=1.2, displace=0.00).setStyles([CYAN,THICK6]).addArrow(0.5,a)

    elif index==2:
        a = Arrow(pos=0.3,size=0.8)
        sm1  = Fermion(in1,center).addLabel(r"\huge \bf SM", pos=-0.1, displace=0.00).setStyles([CYAN,THICK6]).addArrow(0.5,a)
        sm2  = Fermion(in2,center).addLabel(r"\huge \bf SM", pos=-0.1, displace=0.00).setStyles([CYAN,THICK6]).addArrow(0.5,a)

        a = Arrow(pos=0.9,size=0.8)
        dm1  = Fermion(center,out1).addLabel(r"\huge \bf X", pos=1.1, displace=0.00).setStyles([APRICOT,THICK6]).addArrow(0.5,a)
        dm2  = Fermion(center,out2).addLabel(r"\huge \bf X", pos=1.2, displace=0.00).setStyles([APRICOT,THICK6]).addArrow(0.5,a)


    interaction = Circle(center=center, radius=1.5, fill=[pyx.color.cmyk.Green], points = [out_vtx])
    interaction = Circle(center=center, radius=1.5, fill=[pyx.color.cmyk.Green], points = [out_vtx])
        
    time  = Fermion(Point(-3.5,-3.9),Point(3.5,-3.9)).addLabel(r"\large Time", pos=0.5, displace=-0.10).addArrow(1.0)

    ##############################################################################
    name = "%s_%d.pdf" % (outfilename,index)
    fd.draw(name)



#################################################################
#################################################################
def dm_decays(outfilename, index):

    corners = []
    corners.append(Point(-4.2,-2.2))
    corners.append(Point(-4.2,2.2))
    corners.append(Point(4.2,-2.2))
    corners.append(Point(4.2,2.2))

    processOptions()
    fd = FeynDiagram()

    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))
        #define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

    in1 = Point(-1,0)
    in2 = Point(1,0)
    out1 = Point(-4, 0)
    out2 = Point(4,0)

    center = Point(0,0)

    out_vtx = Vertex(-1, 1, mark=CIRCLE)

    decay1 = Point(3,-1)
    decay2 = Point(3,-2)

    jet_len = 2.2

    if index==0:
        interaction = Circle(center=center, radius=0.48, fill=[pyx.color.cmyk.Apricot], points = [out_vtx]).addLabel(r"\huge \bf $\chi$",displace=0.8,angle=90)

    elif index==1:
        interaction = Circle(center=center, radius=0.48, fill=[CROSSHATCHED45,APRICOT], points = [out_vtx]).setStrokeStyle(pyx.style.linestyle.dashed).addLabel(r"\huge \bf $\chi$",displace=0.8,angle=90).setFillStyle(CROSSHATCHED45)

        a = Arrow(pos=1.00,size=0.2)
        dm1  = Fermion(in1,out1).addLabel(r"\huge \bf $e^+$", pos=0.82, displace=0.20).setStyles([CYAN,THICK3]).addArrow(0.5,a)
        dm2  = Fermion(in2,out2).addLabel(r"\huge \bf $e^-$", pos=0.9, displace=- 0.20).setStyles([CYAN,THICK3]).addArrow(0.5,a)

    ##############################################################################
    name = "%s_%d.pdf" % (outfilename,index)
    fd.draw(name)



#################################################################
