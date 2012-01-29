from pyfeyn.user import *

corners = []
corners.append(Point(-3.5,2.5))
corners.append(Point(-3.5,2.5))
corners.append(Point(3.5,-2.5))
corners.append(Point(3.5,2.5))

#################################################################
def neutron_decay(outfilename, index=0):

    processOptions()
    fd = FeynDiagram()

    ############################################################################
    # Make border
    ############################################################################
    define_border = []
    for item in corners:
        define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))

    q0in = Point(-3, 0)

    vtx0 = Vertex(0.0, 0)

    q0out = Point(2.7, +2)
    q1out = Point(2.7, -1.0)
    q2out = Point(2.7, -2)

    f0 = Fermion(q0in, vtx0).addLabel(r"$n$",pos=-0.10,displace=+0.01).addArrow()

    f1 = Fermion(vtx0, q0out).addLabel(r"$p$",pos=1.10,displace=-0.01).addArrow()
    f2 = Fermion(vtx0, q1out).addLabel(r"$e^-$",pos=1.10,displace=-0.01).addArrow()
    f3 = Higgs(vtx0, q2out).addLabel(r"$\bar{\nu}_e$",pos=1.10,displace=-0.01).addArrow()

    ############################################################################
    # Write the output file
    ############################################################################
    fd.draw(outfilename + ".pdf")



################################################################################
def neutron_decay_quark_lines(outfilename, index=0):
  from pyfeyn.user import *

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))
  ###############

  in1 = Point(-3, -2)
  in2 = Point(-3, -1.5)
  in3 = Point(-3, -1)
  out1 = Point(3, -2)
  out2 = Point(3, -1.5)
  out3 = Point(3, -1)

  w_start = Vertex(0, -1, mark=CIRCLE)
  w_decay = Vertex(1.5, 1, mark=CIRCLE)

  #c1 = Circle(center=w_start, radius=0.2, fill=[pyx.color.cmyk.Yellow], points = [w_start])
  #c2 = Circle(center=w_decay, radius=0.2, fill=[pyx.color.rgb.green], points = [w_decay])

  decay1 = Point(3,1)
  decay2 = Point(3,2)

  fa1 = Fermion(in1, out1).addLabel(r"$d$", pos=-0.05, displace=0.01).addLabel(r"$u$", pos=1.05, displace=0.00)
  fa2 = Fermion(in2, out2).addLabel(r"$d$", pos=-0.05, displace=0.01).addLabel(r"$d$", pos=1.05, displace=0.00)
  fa3 = Fermion(in3, out3).addLabel(r"$u$", pos=-0.05, displace=0.01).addLabel(r"$u$", pos=1.05, displace=0.00)

  fx  = Photon(w_start, w_decay).addLabel(r"X", pos=0.70 , displace=-0.16)
  fx_decay2  = Fermion(w_decay, decay2).addLabel(r"\Pep", pos=1.28, displace=0.02)


  name = "%s_%d.pdf" % (outfilename,index)
  fd.draw(name)
