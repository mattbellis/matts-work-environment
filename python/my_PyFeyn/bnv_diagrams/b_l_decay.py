from pyfeyn.user import *

corners = []
corners.append(Point(-4.5,-3.0))
corners.append(Point(-4.5,2.0))
corners.append(Point(4.5,-3.0))
corners.append(Point(4.5,2.0))

#################################################################
def meson_to_baryon_lepton(outfilename, mesonname, q_m0, q_m1, q_b1, q_b2, lep, baryonname, doEllipse=0):
#  from pyfeyn.user import *

  print "Here: " + mesonname + " " + baryonname

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

  if doEllipse:
    reference0 = Ellipse(x=3.3, y=0.2, xradius=0.2, yradius=1.5,  stroke=[pyx.color.rgb.black] , fill=[pyx.color.transparency(1.0)] )
    reference0.setDepth(0.1)

  in1 = Point(-3, 1)
  in2 = Point(-3, 0)
  out1 = Point(3, 1)
  out2 = Point(3, 0)

  #in_vtx = Vertex(3, 1, mark=CIRCLE)
  out_vtx = Vertex(0, 0, mark=CIRCLE)

  xdecay_vtx = Vertex(1.5, -1.5, mark=CIRCLE)

  c1 = Circle(center=out_vtx, radius=0.2, fill=[color.rgb.red], points = [out_vtx])
  c2 = Circle(center=xdecay_vtx, radius=0.2, fill=[color.rgb.blue], points = [xdecay_vtx])

  decay1 = Point(3,-1)
  decay2 = Point(3,-2)

  #l1 = Label("Drell-Yan QCD vertex correction", x=0, y=2)
  #l1 = Label("\PcgLp", x=4, y=0.2)
  l1 = Label("\P"+mesonname, x=-4.0, y=0.5)
  l2 = Label("\P"+baryonname, x=4, y=0.2)


  #fa1 = Fermion(in1, out1).addLabel(r"\Pqu", pos=-0.05, displace=0.01).addLabel(r"\Pqu", pos=1.05, displace=0.00)
  fa1 = Fermion(in1, out1).addLabel(r"\P"+q_m0, pos=-0.05, displace=0.01).addLabel(r"\P"+q_m0, pos=1.05, displace=0.00)
  fa2 = Fermion(in2, out2).addLabel(r"\P"+q_m1, pos=-0.05, displace=0.01).addLabel(r"\P"+q_b1, pos=1.05, displace=0.00)
  fx  = Photon(out_vtx, xdecay_vtx).addLabel(r"X")
  fx_decay1  = Fermion(xdecay_vtx, decay1).addLabel(r"\P"+q_b2, pos=1.23, displace=0.00)
  fx_decay2  = Fermion(xdecay_vtx, decay2).addLabel(r"\P"+lep, pos=1.28, displace=0.02)



  fd.draw(outfilename + ".pdf")



#################################################################

def baryon_to_meson_lepton(outfilename, mesonname, q_m0, q_m1, q_b1, q_b2, baryonname, doEllipse=0):
  from pyfeyn.user import *

  print "Here: " + mesonname + " " + baryonname

  processOptions()
  fd = FeynDiagram()

  ####### Set common borders
  #corners = []
  #corners.append(Point(-4.5,-3.5))
  #corners.append(Point(-4.5,3.5))
  #corners.append(Point(4.5,-3.5))
  #corners.append(Point(4.5,3.5))

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))
  ###############

  in1 = Point(-3, 1)
  in2 = Point(-3, 0)
  in3 = Point(-3, -1)
  out1 = Point(3, 1)
  out2 = Point(3, 0)
  out3 = Point(3, -1)

  out_vtx = Vertex(0, 0, mark=CIRCLE)
  xdecay_vtx = Vertex(1.5, -1, mark=CIRCLE)

  c1 = Circle(center=out_vtx, radius=0.2, fill=[color.rgb.red], points = [out_vtx])
  c2 = Circle(center=xdecay_vtx, radius=0.2, fill=[color.rgb.blue], points = [xdecay_vtx])

  decay1 = Point(3,-1)
  decay2 = Point(3,-2)

  #l1 = Label("Drell-Yan QCD vertex correction", x=0, y=2)
  #l1 = Label("\PcgLp", x=4, y=0.2)
  l1 = Label("\P"+mesonname, x=-4.0, y=0.0)
  l2 = Label("\P"+baryonname, x=4, y=0.7)


  #fa1 = Fermion(in1, out1).addLabel(r"\Pqu", pos=-0.05, displace=0.01).addLabel(r"\Pqu", pos=1.05, displace=0.00)
  fa1 = Fermion(in1, out1).addLabel(r"\P"+q_m0, pos=-0.05, displace=0.01).addLabel(r"\P"+q_m0, pos=1.05, displace=0.00)
  fa2 = Fermion(in2, out2).addLabel(r"\P"+q_m1, pos=-0.05, displace=0.01).addLabel(r"\P"+q_b1, pos=1.05, displace=0.00)
  fa3 = Fermion(in3, xdecay_vtx).addLabel(r"\Pqu", pos=-0.05, displace=0.01)
  fx  = Photon(out_vtx, xdecay_vtx).addLabel(r"X", pos=0.70 , displace=-0.16)
  #fx_decay1  = Fermion(xdecay_vtx, decay1).addLabel(r"\P"+q_b2, pos=1.23, displace=0.00)
  fx_decay2  = Fermion(xdecay_vtx, decay2).addLabel(r"\Pep", pos=1.28, displace=0.02)


  if doEllipse:
    reference0 = Ellipse(x=3.3, y=0.5, xradius=0.2, yradius=1.0,  stroke=[pyx.color.rgb.black] , fill=[pyx.color.transparency(1.0)] )
    reference0.setDepth(0.1)


  fd.draw(outfilename + ".pdf")
