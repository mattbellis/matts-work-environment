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

  in1 = Point(-2.5, -2)
  in2 = Point(-2.5, -1.5)
  in3 = Point(-2.5, -1)
  out1 = Point(2.5, -2)
  out2 = Point(2.5, -1.5)
  out3 = Point(2.5, -1)

  outhalf1 = Point(0, -2)
  outhalf2 = Point(0, -1.5)
  outhalf3 = Point(0, -1)

  w_start = Vertex(0, -1)
  w_decay = Vertex(1.2, 1)

  decay1 = Point(2.5,0.5)
  decay2 = Point(2.5,2)

  if index<1:
      fhalfa1 = Fermion(in1, outhalf1).addLabel(r"$u$", pos=-0.05, displace=0.01).addLabel(r"$u$", pos=1.05, displace=0.00)
      fhalfa2 = Fermion(in2, outhalf2).addLabel(r"$d$", pos=-0.05, displace=0.01).addLabel(r"$d$", pos=1.05, displace=0.00).addLabel(r"{\Large $n$}", pos=-0.30, displace=0.10)
      fhalfa3 = Fermion(in3, outhalf3).addLabel(r"$d$", pos=-0.05, displace=0.01).addLabel(r"$d$", pos=1.05, displace=0.00)

  if index>=1:
      c1 = Circle(center=w_start, radius=0.1, fill=[pyx.color.cmyk.Yellow], points = [w_start])
      fx  = Photon(w_start, w_decay).addLabel(r"$W^-$", pos=0.70 , displace=-0.35)

      fa1 = Fermion(in1, out1).addLabel(r"$u$", pos=-0.05, displace=0.01).addLabel(r"$u$", pos=1.05, displace=0.00)
      fa2 = Fermion(in2, out2).addLabel(r"$d$", pos=-0.05, displace=0.01).addLabel(r"$d$", pos=1.05, displace=0.00).addLabel(r"{\Large $n$}", pos=-0.15, displace=0.10)
      fa3 = Fermion(in3, out3).addLabel(r"$d$", pos=-0.05, displace=0.01).addLabel(r"$u$", pos=1.05, displace=0.00).addLabel(r"{\Large $p$}", pos=1.15, displace=0.10)

  if index>=2:
      c2 = Circle(center=w_decay, radius=0.1, fill=[pyx.color.cmyk.Yellow], points = [w_decay])
      fx_decay1  = Fermion(w_decay, decay1).addLabel(r"$e^-$", pos=1.28, displace=0.02)
      fx_decay2  = Higgs(w_decay, decay2).addLabel(r"$\bar{\nu}_e$", pos=1.28, displace=0.02)


  name = "%s_%d.pdf" % (outfilename,index)
  fd.draw(name)

################################################################################
def antineutron_decay_quark_lines(outfilename, index=0):
  from pyfeyn.user import *

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))
  ###############

  in1 = Point(-2.5, -2)
  in2 = Point(-2.5, -1.5)
  in3 = Point(-2.5, -1)
  out1 = Point(2.5, -2)
  out2 = Point(2.5, -1.5)
  out3 = Point(2.5, -1)

  outhalf1 = Point(0, -2)
  outhalf2 = Point(0, -1.5)
  outhalf3 = Point(0, -1)

  w_start = Vertex(0, -1)
  w_decay = Vertex(1.2, 1)

  decay1 = Point(2.5,0.5)
  decay2 = Point(2.5,2)

  if index<1:
      fhalfa1 = Fermion(in1, outhalf1).addLabel(r"$\bar{u}$", pos=-0.05, displace=0.01).addLabel(r"$\bar{u}$", pos=1.05, displace=0.00)
      fhalfa2 = Fermion(in2, outhalf2).addLabel(r"$\bar{d}$", pos=-0.05, displace=0.01).addLabel(r"$\bar{d}$", pos=1.05, displace=0.00).addLabel(r"{\Large $\bar{n}$}", pos=-0.30, displace=0.10)
      fhalfa3 = Fermion(in3, outhalf3).addLabel(r"$\bar{d}$", pos=-0.05, displace=0.01).addLabel(r"$\bar{d}$", pos=1.05, displace=0.00)

  if index>=1:
      c1 = Circle(center=w_start, radius=0.1, fill=[pyx.color.cmyk.Yellow], points = [w_start])
      fx  = Photon(w_start, w_decay).addLabel(r"$W^+$", pos=0.70 , displace=-0.35)

      fa1 = Fermion(in1, out1).addLabel(r"$\bar{u}$", pos=-0.05, displace=0.01).addLabel(r"$\bar{u}$", pos=1.05, displace=0.00)
      fa2 = Fermion(in2, out2).addLabel(r"$\bar{d}$", pos=-0.05, displace=0.01).addLabel(r"$\bar{d}$", pos=1.05, displace=0.00).addLabel(r"{\Large $\bar{n}$}", pos=-0.15, displace=0.10)
      fa3 = Fermion(in3, out3).addLabel(r"$\bar{d}$", pos=-0.05, displace=0.01).addLabel(r"$\bar{u}$", pos=1.05, displace=0.00).addLabel(r"{\Large $\bar{p}$}", pos=1.15, displace=0.10)

  if index>=2:
      c2 = Circle(center=w_decay, radius=0.1, fill=[pyx.color.cmyk.Yellow], points = [w_decay])
      fx_decay1  = Fermion(w_decay, decay1).addLabel(r"$e^+$", pos=1.28, displace=0.02)
      fx_decay2  = Higgs(w_decay, decay2).addLabel(r"$\nu_e$", pos=1.28, displace=0.02)


  name = "%s_%d.pdf" % (outfilename,index)
  fd.draw(name)
