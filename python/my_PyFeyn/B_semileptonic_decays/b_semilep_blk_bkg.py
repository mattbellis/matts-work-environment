from pyfeyn.user import *

corners = []
corners.append(Point(-4.5,-3.0))
corners.append(Point(-4.5,2.0))
corners.append(Point(4.5,-3.0))
corners.append(Point(4.5,2.0))

#################################################################
def b_semilep(outfilename, index):

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))

  if index==2:
    reference0 = Ellipse(x=3.3,y=0.2,xradius=0.2,yradius=1.5,stroke=[pyx.color.rgb.white],fill=[pyx.color.rgb.black])
    reference0.setDepth(0.0)

  in1 = Point(-3, 1)
  in2 = Point(-3, 0)
  out1 = Point(3, 1)
  out2 = Point(3, 0)

  out_vtx = Vertex(0, 0, mark=CIRCLE)

  xdecay_vtx = Vertex(1.5, -1.5, mark=CIRCLE)

  c1 = Circle(center=out_vtx, radius=0.1, fill=[pyx.color.cmyk.Yellow], points = [out_vtx])
  c2 = Circle(center=xdecay_vtx, radius=0.1, fill=[color.cmyk.Yellow], points = [xdecay_vtx])

  decay1 = Point(3,-1)
  decay2 = Point(3,-2)

  if index==0:
      l1 = Label(r"{\Large $B^0$}", x=-4.0, y=0.5)
      l2 = Label(r"{\Large $D^-$}", x=4, y=0.6)
  elif index==1:
      l1 = Label(r"{\Large $\bar{B}^0$}", x=-4.0, y=0.5)
      l2 = Label(r"{\Large $\bar{D}^+$}", x=4, y=0.6)

  if index==0:
      fa1 = Fermion(in1, out1).addLabel(r"$\bar{b}$", pos=-0.05, displace=0.01).addLabel(r"$\bar{c}$", pos=1.05, displace=0.00)
      fa2 = Fermion(in2, out2).addLabel(r"$d$", pos=-0.05, displace=0.01).addLabel(r"$d$", pos=1.05, displace=0.00)
      fx  = Photon(out_vtx, xdecay_vtx).addLabel(r"$W^+$",pos=0.5,displace=0.4)
      fx_decay1  = Fermion(xdecay_vtx, decay1).addLabel(r"$\mu^+$", pos=1.23, displace=0.00)
      fx_decay2  = Higgs(xdecay_vtx, decay2).addLabel(r"$\nu_{\mu}$", pos=1.28, displace=0.02)
  elif index==1:
      fa1 = Fermion(in1, out1).addLabel(r"$b$", pos=-0.05, displace=0.01).addLabel(r"$c$", pos=1.05, displace=0.00)
      fa2 = Fermion(in2, out2).addLabel(r"$\bar{d}$", pos=-0.05, displace=0.01).addLabel(r"$\bar{d}$", pos=1.05, displace=0.00)
      fx  = Photon(out_vtx, xdecay_vtx).addLabel(r"$W^-$",pos=0.5,displace=0.4)
      fx_decay1  = Fermion(xdecay_vtx, decay1).addLabel(r"$\mu^-$", pos=1.23, displace=0.00)
      fx_decay2  = Higgs(xdecay_vtx, decay2).addLabel(r"$\bar{\nu}_{\mu}$", pos=1.28, displace=0.02)

  name = "%s_%d.pdf" % (outfilename,index)
  fd.draw(name)



#################################################################
