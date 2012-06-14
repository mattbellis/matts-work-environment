#! /usr/bin/env python

from pyfeyn.user import *

processOptions()
fd = FeynDiagram()

reference0 = Ellipse(x=3, y=1, xradius=0.3, yradius=1.5,  stroke = [pyx.color.rgb.black] )

in1 = Point(-3, 1)
in2 = Point(-3, 0)
out1 = Point(3, 1)
out2 = Point(3, 0)

in_vtx = Vertex(3, 1, mark=CIRCLE)
out_vtx = Vertex(0, 0, mark=CIRCLE)

xdecay_vtx = Vertex(1.5, -1.5, mark=CIRCLE)

c1 = Circle(center=out_vtx, radius=0.2, fill=[color.rgb.red], points = [out_vtx])
c2 = Circle(center=xdecay_vtx, radius=0.2, fill=[color.rgb.blue], points = [xdecay_vtx])

decay1 = Point(3,-1)
decay2 = Point(3,-2)

#l1 = Label("Drell-Yan QCD vertex correction", x=0, y=2)
l1 = Label("\PcgLp", x=4, y=0.2)


fa1 = Fermion(in1, out1).addLabel(r"\Pqu", pos=-0.05, displace=0.01).addLabel(r"\Pqu", pos=1.05, displace=0.00)
fa2 = Fermion(in2, out2).addLabel(r"\Paqb", pos=-0.05, displace=0.01).addLabel(r"\Pqc", pos=1.05, displace=0.00)
fx  = Higgs(out_vtx, xdecay_vtx).addLabel(r"X")
fx_decay1  = Fermion(xdecay_vtx, decay1).addLabel(r"\Pqd", pos=1.23, displace=0.00)
fx_decay2  = Fermion(xdecay_vtx, decay2).addLabel(r"\Plm", pos=1.28, displace=0.02)



fd.draw("test.pdf")
