from pyfeyn.user import *

corners = []
corners.append(Point(-3.0,-2.5))
corners.append(Point(-3.0,2.5))
corners.append(Point(3.0,-2.5))
corners.append(Point(3.0,2.5))

#################################################################
def decay(outfilename, option=0):
#  from pyfeyn.user import *

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

  n_pt = Point(1.75, 1.25)
  p_pt = Point(0,0)
  e_pt = Point(-2,-1)
  nrino_pt = Point(-1.5,-1.5)

  proton = Circle(center=p_pt, radius=0.7, fill=[CYAN]).addLabel(r"\Pp", displace=1.0, size=pyx.text.size.Huge)
  if option!=0:
    neutron = Circle(center=n_pt, radius=0.7, fill=[GREEN])
    electron = Circle(center=e_pt, radius=0.3, fill=[RED])
    neutrino = Circle(center=nrino_pt, radius=0.1, fill=[color.rgb.black])

    d1 =     Higgs(p_pt, n_pt).addArrow(0.60).addLabel(r"\Pn", pos=1.40, displace=0.5)
    d2 =     Higgs(p_pt, e_pt).addArrow(0.80).addLabel(r"\Pep", pos=1.10, displace=0.35)
    d3 =     Higgs(p_pt, nrino_pt).addArrow(0.80).addLabel(r"\Pgne", pos=1.10, displace=-0.25)

  #decay1 = Point(0.0, 0.0)

  #l1 = Label("\P"+mesonname, x=-4.0, y=0.5)
  #l2 = Label("\P"+baryonname, x=4, y=0.2)


  #gamma = Photon(in1, out1).addArrow().addLabel(r"\Pgg", pos=-0.05, displace=0.35)
  #pion =     Fermion(in1, decay1).addArrow().addLabel(r"\Ppi", pos=-0.05, displace=-0.15)

  #d1 =     Fermion(decay1, out1).addArrow().addLabel(r"\Pe", pos=0.95, displace=-0.15)
  #d2 =     Fermion(out2, decay1).addArrow().addLabel(r"\Pnu", pos=-0.05, displace=-0.15)

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

