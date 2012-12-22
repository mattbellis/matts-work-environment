from pyfeyn.user import *

corners = []
corners.append(Point(-4.5,-2.5))
corners.append(Point(-4.5,3.5))
corners.append(Point(4.5,-2.5))
corners.append(Point(4.5,3.5))

#################################################################
def piondecay(outfilename, option=0):
#  from pyfeyn.user import *

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

  in1 = Point(-4, 0)
  out1 = Point(4, 2)
  out2 = Point(4, -2)

  #xdecay_vtx = Vertex(1.5, -1.5, mark=CIRCLE)

  decay1 = Point(0.0, 0.0)

  #l1 = Label("\P"+mesonname, x=-4.0, y=0.5)
  #l2 = Label("\P"+baryonname, x=4, y=0.2)


  #gamma = Photon(in1, out1).addArrow().addLabel(r"\Pgg", pos=-0.05, displace=0.35)
  pion =     Fermion(in1, decay1).addArrow().addLabel(r"\Ppi", pos=-0.05, displace=-0.15)

  d1 =     Fermion(decay1, out1).addArrow().addLabel(r"\Pe", pos=0.95, displace=-0.15)
  d2 =     Fermion(out2, decay1).addArrow().addLabel(r"\Pnu", pos=-0.05, displace=-0.15)

  #if option==0:
    #isobar.setStyles([color.rgb.red,THICK5])

  #elif option==1:
    #apt1 = Point(1.5,0.0)
    #apt2 = Point(-1.0,-1.0)
    #anglecm =     Higgs(apt1, apt2).bend(-0.5)
    #anglecm.setStyles([color.rgb.red,THICK3]).addArrow()

  #elif option==2:
    #apt1 = Point(3.3,1.3)
    #apt2 = Point(2.7,2.7)
    #anglecm =     Higgs(apt1, apt2).bend(0.2)
    #anglecm.setStyles([color.rgb.red,THICK3]).addArrow()

  fd.draw(outfilename + ".pdf")



#################################################################

