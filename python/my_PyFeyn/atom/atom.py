from pyfeyn.user import *
from math import *

corners = []
corners.append(Point(-3.0,-3.0))
corners.append(Point(-3.0,3.0))
corners.append(Point(3.0,-3.0))
corners.append(Point(3.0,3.0))

#################################################################
def atom(outfilename, option=0):
#  from pyfeyn.user import *

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.white], fill=[color.rgb.white]))

  n_str = ["p","n","p","n"]
  #n_angle = [20,110,200,-70]
  n_angle = [20,100,200,-70]
  n_color = [WHITE,GREY,WHITE,GREY]
  nl_ang = [0,120,220,300]
  n_x = []
  n_y = []
  n_dis = 0.5
  n_r = 0.5


  nucleons = []
  for i,ang in enumerate(n_angle):
      a = (ang/360.0)*2*3.14159
      n_x.append(n_dis*cos(a))
      n_y.append(n_dis*sin(a))
      n_pt = Point(n_x[i],n_y[i])
      if option==0:
          nucleons.append(Circle(center=n_pt, radius=n_r, fill=[WHITE]))
      elif option==1 or option==2:
          nucleons.append(Circle(center=n_pt, radius=n_r, fill=[n_color[i]]).addLabel(n_str[i], displace=0.3, angle=nl_ang[i]))
      else:
          nucleons.append(Circle(center=n_pt, radius=n_r, fill=[n_color[i]]))

  if option>1:
      vtx1 = Point(-2.5,0)
      vtx2 = Point( 2.5,0)
      elec1_path = Higgs(vtx1, vtx2).bend(-2.5)
      elec1_path = Higgs(vtx2, vtx1).bend(-2.5)

      electrons = []
      e_pt = []
      e_pt.append(Point(0,2.5))
      e_pt.append(Point(0,-2.5))
      el_ang = [90, -90]
      for i,ep in enumerate(e_pt):
          electrons.append(Circle(center=ep, radius=0.1, fill=[MELON]).addLabel(r"\Pe", displace=0.3, angle=el_ang[i]))


  q = []
  q_angle = [20, 40, 20, 130]
  q_color = [RED,BLUE,GREEN]
  for i,x in enumerate(n_x):
      y = n_y[i]
      for j,k in enumerate(q_color):
          ang = q_angle[i] + 120*j
          a = (ang/360.0)*2*3.14159
          qx = x + 0.25*cos(a)
          qy = y + 0.25*sin(a)
          q_pt = Point(qx,qy)
          if option>2:
              q.append(Circle(center=q_pt, radius=0.2, fill=[q_color[j]]).addLabel(r"\Pq",displace=0.01))
      

  outfilename = "%s_%d.pdf" % (outfilename,option)
  fd.draw(outfilename)



#################################################################

