from pyfeyn.user import *

corners = []
corners.append(Point(-4.0,-2.0))
corners.append(Point(-4.0,2.0))
corners.append(Point(4.0,-2.0))
corners.append(Point(4.0,2.0))

#################################################################
def std_mod(outfilename, option=0):
#  from pyfeyn.user import *

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

  q_pts = []
  q_pts.append(Point(-3.0,1.5))
  q_pts.append(Point(-3.0,0.0))
  q_pts.append(Point(0.0,0.0))
  q_pts.append(Point(0.0,1.5))
  q_pts.append(Point(1.5,-1.5))
  q_pts.append(Point(1.5,1.5))

  q_r = [0.03, 0.03, 0.1, 0.3, 5.0, 100.0]
  q_str = ["up","down","strange","charm","bottom","top"]
  q_angle = [90,-90,-90,90,-90,90]


  quarks = []
  for i,qp in enumerate(q_pts):
      quarks.append(Circle(center=qp, radius=q_r[i], fill=[CYAN]).addLabel(q_str[i], displace=0.3, angle=q_angle[i]))


  if option!=0:
    neutron = Circle(center=n_pt, radius=0.7, fill=[GREEN])
    electron = Circle(center=e_pt, radius=0.3, fill=[RED])
    neutrino = Circle(center=nrino_pt, radius=0.1, fill=[color.rgb.black])

    d1 =     Higgs(p_pt, n_pt).addArrow(0.60).addLabel(r"\Pn", pos=1.40, displace=0.5)
    d2 =     Higgs(p_pt, e_pt).addArrow(0.80).addLabel(r"\Pep", pos=1.10, displace=0.35)
    d3 =     Higgs(p_pt, nrino_pt).addArrow(0.80).addLabel(r"\Pgne", pos=1.10, displace=-0.25)


  fd.draw(outfilename + ".pdf")



#################################################################

