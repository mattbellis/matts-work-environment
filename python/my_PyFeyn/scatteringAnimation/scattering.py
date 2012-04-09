from pyfeyn.user import *

xmin = -4.5
xmax = 4.5
ymin = -3.0
ymax = 2.0

xlen = xmax - xmin
ylen = ymax - ymin

corners = []
corners.append(Point(xmin, ymin))
corners.append(Point(xmin, ymax))
corners.append(Point(xmax, ymin))
corners.append(Point(xmax, ymax))

#################################################################
def scattering(outfilename, flightincrement):
#  from pyfeyn.user import *
  q_m0 = "qu"
  q_m1 = "aqd"

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

  xstart = -3.0
  xdistmax = 3.0
  xdist = xstart + flightincrement*(xdistmax-xstart)

  in1 = Point(xstart, 1)
  in2 = Point(xstart, 0)

  out1 = Point(xdist, 1)
  out2 = Point(xdist, 0)

  #fa1 = Fermion(in1, out1).addLabel(r"\Pqu", pos=-0.05, displace=0.01).addLabel(r"\Pqu", pos=1.05, displace=0.00)
  fa1 = Fermion(in1, out1).addLabel(r"\P"+q_m0, pos=-0.05, displace=0.01).addLabel(r"\P"+q_m0, pos=1.05, displace=0.00)
  fa2 = Fermion(in2, out2).addLabel(r"\P"+q_m1, pos=-0.05, displace=0.01).addLabel(r"\P"+q_m1, pos=1.05, displace=0.00)

  fd.draw(outfilename + ".pdf")


#################################################################

