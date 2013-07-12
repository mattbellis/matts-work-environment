import numpy as np
from pyfeyn.user import *

corners = []
corners.append(Point(-4.5,-3.0))
corners.append(Point(-4.5,2.0))
corners.append(Point(4.5,-3.0))
corners.append(Point(4.5,2.0))

#################################################################
def top_decays(outfilename, index):

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
      #define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

  in1 = Point(-4, 1)
  in2 = Point(-4, 0)
  out1 = Point(2, 1)
  out2 = Point(2, 0)

  out_vtx = Vertex(-1, 1, mark=CIRCLE)

  xdecay_vtx = Vertex(1.0, -1.0, mark=CIRCLE)

  c1 = Circle(center=out_vtx, radius=0.1, fill=[pyx.color.cmyk.Yellow], points = [out_vtx])
  c2 = Circle(center=xdecay_vtx, radius=0.1, fill=[color.cmyk.Green], points = [xdecay_vtx])

  decay1 = Point(3,-1)
  decay2 = Point(3,-2)

  jet_len = 2.2

  if index==0 or index==1:
      l1 = Label(r"{\Large $t$}", x=-4.0, y=1.5)
      l2 = Label(r"{\Large $q=(b)$}", x=1, y=1.6)
  elif index==2 or index==3:
      l1 = Label(r"{\Large $t$}", x=-4.0, y=1.5)
      l2 = Label(r"{\Large $q=(\bar{b},\bar{s},\bar{d})$}", x=1, y=1.6)

  ##############################################################################
  if index==0:
      fa2 = Fermion(in1, out1)
      for a in [-0.2,-0.10,0.0,0.10,0.2]:
          xout = jet_len*np.cos(a) + out1.getX()
          yout = jet_len*np.sin(a) + out1.getY()
          jet = Fermion(out1,Point(xout,yout)).setStyles([CYAN,THICK2])
      fx  = Photon(out_vtx, xdecay_vtx).addLabel(r"$W^+$",pos=0.5,displace=0.4)
      fx_decay1  = Fermion(xdecay_vtx, decay1).addLabel(r"$\ell^+$", pos=1.23, displace=0.00).setStyles([APRICOT,THICK2])
      fx_decay2  = Higgs(xdecay_vtx, decay2).addLabel(r"$\nu_{\ell}$", pos=1.28, displace=0.02)
  ##############
  elif index==1:
      jet_decay1 = Point(2,-1)
      jet_decay2 = Point(2,-2)
      fa2 = Fermion(in1, out1)
      for a in [-0.2,-0.10,0.0,0.10,0.2]:
          xout = jet_len*np.cos(a) + out1.getX()
          yout = jet_len*np.sin(a) + out1.getY()
          jet = Fermion(out1,Point(xout,yout)).setStyles([CYAN,THICK2])
      fx  = Photon(out_vtx, xdecay_vtx).addLabel(r"$W^+$",pos=0.5,displace=0.4)
      fx_decay1  = Fermion(xdecay_vtx, jet_decay1).addLabel(r"$q$", pos=1.23, displace=-0.15)
      for a in [-0.2,-0.10,0.0,0.10,0.2]:
          xout = jet_len*np.cos(a) + jet_decay1.getX()
          yout = jet_len*np.sin(a) + jet_decay1.getY()
          jet = Fermion(jet_decay1,Point(xout,yout)).setStyles([CYAN,THICK2])
      fx_decay2  = Fermion(xdecay_vtx, jet_decay2).addLabel(r"$\bar{q}$", pos=1.28, displace=0.02)
      for a in [-0.2,-0.10,0.0,0.10,0.2]:
          xout = jet_len*np.cos(a) + jet_decay2.getX()
          yout = jet_len*np.sin(a) + jet_decay2.getY()
          jet = Fermion(jet_decay2,Point(xout,yout)).setStyles([CYAN,THICK2])
  ##############
  elif index==2:
      jet_decay1 = Point(2,-1)
      jet_decay2 = Point(2,-2)
      fa2 = Fermion(in1, out1)
      for a in [-0.2,-0.10,0.0,0.10,0.2]:
          xout = jet_len*np.cos(a) + out1.getX()
          yout = jet_len*np.sin(a) + out1.getY()
          jet = Fermion(out1,Point(xout,yout)).setStyles([CYAN,THICK2])
      fx  = Photon(out_vtx, xdecay_vtx).addLabel(r"$X^+ (q=+\frac{1}{3})$",pos=0.5,displace=0.9)
      fx_decay1  = Fermion(xdecay_vtx, decay1).addLabel(r"$\ell^+$", pos=1.23, displace=0.00).setStyles([APRICOT,THICK2])
      fx_decay2  = Fermion(xdecay_vtx, jet_decay2).addLabel(r"$\bar{q}=(\bar{c},\bar{u})$", pos=1.28, displace=0.02)
      for a in [-0.2,-0.10,0.0,0.10,0.2]:
          xout = jet_len*np.cos(a) + jet_decay2.getX()
          yout = jet_len*np.sin(a) + jet_decay2.getY()
          jet = Fermion(jet_decay2,Point(xout,yout)).setStyles([CYAN,THICK2])

  ##############
  elif index==3:
      jet_decay1 = Point(2,-1)
      jet_decay2 = Point(2,-2)
      fa2 = Fermion(in1, out1)
      for a in [-0.2,-0.10,0.0,0.10,0.2]:
          xout = jet_len*np.cos(a) + out1.getX()
          yout = jet_len*np.sin(a) + out1.getY()
          jet = Fermion(out1,Point(xout,yout)).setStyles([CYAN,THICK2])
      fx  = Photon(out_vtx, xdecay_vtx).addLabel(r"$X^+ (q=+\frac{1}{3})$",pos=0.5,displace=0.9)
      fx_decay1  = Higgs(xdecay_vtx, decay1).addLabel(r"$\nu$", pos=1.23, displace=0.00)
      fx_decay2  = Fermion(xdecay_vtx, jet_decay2).addLabel(r"$\bar{q}=(\bar{b},\bar{s},\bar{d})$", pos=1.28, displace=0.02)
      for a in [-0.2,-0.10,0.0,0.10,0.2]:
          xout = jet_len*np.cos(a) + jet_decay2.getX()
          yout = jet_len*np.sin(a) + jet_decay2.getY()
          jet = Fermion(jet_decay2,Point(xout,yout)).setStyles([CYAN,THICK2])


  ##############################################################################
  name = "%s_%d.pdf" % (outfilename,index)
  fd.draw(name)



#################################################################
