from pyfeyn.user import *

corners = []
corners.append(Point(-4.5,-2.5))
corners.append(Point(-4.5,3.5))
corners.append(Point(4.5,-2.5))
corners.append(Point(4.5,3.5))

#################################################################
def twopi_2bodymass(outfilename, option=0):
#  from pyfeyn.user import *

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

  in1 = Point(-4, 0)
  in2 = Point( 4, 0)
  out1 = Point(0, 0)
  out2 = Point(0, 0)

  #xdecay_vtx = Vertex(1.5, -1.5, mark=CIRCLE)

  decay1 = Point(2.0,1.5)
  decay2 = Point(-2.5,-2)

  d1_pt = Point(3.2, 3.2)
  d2_pt = Point(4.0, 1.0)

  #l1 = Label("\P"+mesonname, x=-4.0, y=0.5)
  #l2 = Label("\P"+baryonname, x=4, y=0.2)


  gamma = Photon(in1, out1).addArrow().addLabel(r"\Pgg", pos=-0.05, displace=0.35)
  p_i =     Fermion(in2, out2).addArrow().addLabel(r"\Pproton", pos=-0.05, displace=-0.15)
  isobar =     Fermion(out2, decay1)

  recoil =     Fermion(out2, decay2).addArrow()
  d1 =     Fermion(decay1, d1_pt).addArrow()
  d2 =     Fermion(decay1, d2_pt).addArrow()

  if option==0:
    isobar.setStyles([color.rgb.red,THICK5])

  elif option==1:
    apt1 = Point(1.5,0.0)
    apt2 = Point(-1.0,-1.0)
    anglecm =     Higgs(apt1, apt2).bend(-0.5)
    anglecm.setStyles([color.rgb.red,THICK3]).addArrow()

  elif option==2:
    apt1 = Point(3.3,1.3)
    apt2 = Point(2.7,2.7)
    anglecm =     Higgs(apt1, apt2).bend(0.2)
    anglecm.setStyles([color.rgb.red,THICK3]).addArrow()

  fd.draw(outfilename + ".pdf")



#################################################################

