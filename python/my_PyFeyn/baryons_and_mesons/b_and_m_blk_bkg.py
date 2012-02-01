from pyfeyn.user import *
from math import *
from pyx import *

corners = []
corners.append(Point(-4.5,-4.5))
corners.append(Point(-4.5,4.5))
corners.append(Point(4.5,-4.5))
corners.append(Point(4.5,4.5))

#################################################################
def baryon(outfilename, option=0):
#  from pyfeyn.user import *

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))

  n_angle = [110,80,130,30]
  n_color = [WHITE,GREY,WHITE,GREY]
  n_x = []
  n_y = []
  n_dis = [6.0,5.5,3.0,1.0]
  n_r = [0.7,0.7,1.2,1.5]
  x0 = 0.0
  y0 = -3.0

  n_str = []
  n_str.append(['p',['qu','qu','qd']])
  n_str.append(['n',['qu','qd','qd']])
  n_str.append(['gL',['qu','qd','qs']])
  n_str.append(['cgLp',['qu','qd','qc']])



  q_rad = []
  q_rad.append([0.3,0.3,0.3])
  q_rad.append([0.3,0.3,0.3])
  q_rad.append([0.3,0.3,0.6])
  q_rad.append([0.3,0.3,1.0])

  q_dis = []
  q_dis.append([0.35,0.35,0.35])
  q_dis.append([0.35,0.35,0.35])
  q_dis.append([0.75,0.75,0.35])
  q_dis.append([1.05,1.05,0.35])

  q_angle = [90, 90, 90, 90]
  q_color = [RED,CYAN,SPRINGGREEN]
  nucleons = []
  q = []
  for i in range(0,option):
      ang = n_angle[i]
      a = (ang/360.0)*2*3.14159
      n_x.append(x0+n_dis[i]*cos(a))
      n_y.append(y0+n_dis[i]*sin(a))
      n_pt = Point(n_x[i],n_y[i])
      nucleons.append(Circle(center=n_pt, radius=n_r[i], fill=[GREY]).addLabel(r"\P"+n_str[i][0],displace=n_r[i]+0.3,angle=0,size=pyx.text.size.Large))

      # Quarks
      x = n_x[i]
      y = n_y[i]
      for j,k in enumerate(q_color):
          ang = q_angle[i] + 120*j
          a = (ang/360.0)*2*3.14159
          qx = x + q_dis[i][j]*cos(a)
          qy = y + q_dis[i][j]*sin(a)
          q_pt = Point(qx,qy)
          if option>=0:
              #q.append(Circle(center=q_pt, radius=0.3, fill=[q_color[j]]).addLabel(r"\Pq",displace=0.01,size=pyx.text.size.Large))
              print n_str[i][1][j]
              q.append(Circle(center=q_pt, radius=q_rad[i][j], fill=[q_color[j]]).addLabel(r"\P"+n_str[i][1][j],displace=0.01,size=pyx.text.size.Large))
      

  outfilename = "%s_%d.pdf" % (outfilename,option)
  fd.draw(outfilename)

#################################################################

#################################################################
def meson(outfilename, option=0):
#  from pyfeyn.user import *

  processOptions()
  fd = FeynDiagram()

  define_border = []
  for item in corners:
    define_border.append(Circle(center=item, radius=0.01, stroke=[color.rgb.black], fill=[color.rgb.black]))

  n_angle = [110,80,115,30]
  n_color = [WHITE,GREY,WHITE,GREY]
  n_x = []
  n_y = []
  n_dis = [6.0,5.5,3.9,1.0]
  n_r = [0.6,0.6,1.2,1.5]
  x0 = 0.0
  y0 = -3.0

  n_str = []
  n_str.append(['{\Huge $\pi^+$}',['{\Large $u$}','{\Large $\\bar{d}$}']])
  n_str.append(['{\Huge $\pi^+$}',['{\Large $d$}','{\Large $\\bar{u}$}']])
  n_str.append(['{\Huge $J/\psi$}',['{\Large $c$}','{\Large $\\bar{c}$}']])
  n_str.append(['{\Huge $B^-$}',['{\Large $b$}','{\Large $\\bar{d}$}']])



  q_rad = []
  q_rad.append([0.5,0.5])
  q_rad.append([0.5,0.5])
  q_rad.append([0.6,0.6])
  q_rad.append([0.75,0.5])

  q_dis = []
  q_dis.append([0.55,0.55])
  q_dis.append([0.55,0.55])
  q_dis.append([0.65,0.65])
  q_dis.append([0.75,0.55])

  '''
  if option==0:
      col = color.cmyk.White
      #text.set(mode="latex")
      text.preamble(r"\usepackage{color}")
      text.preamble(r"\definecolor{COL}{cmyk}{%(c)g,%(m)g,%(y)g,%(k)g}" % col.color)
  '''

  q_angle = [90, 90, 90, 90]
  q_color = [CYAN,CYAN]
  mesons = []
  q = []
  aq = []
  for i in range(0,option):
      ang = n_angle[i]
      a = (ang/360.0)*2*3.14159
      n_x.append(x0+n_dis[i]*cos(a))
      n_y.append(y0+n_dis[i]*sin(a))
      n_pt = Point(n_x[i],n_y[i])
      if i==2:
          mesons.append(Circle(center=n_pt, radius=n_r[i], fill=[TAN]).addLabel(n_str[i][0],displace=n_r[i]+0.8,angle=225,size=pyx.text.size.large))
      else:
          mesons.append(Circle(center=n_pt, radius=n_r[i], fill=[TAN]).addLabel(n_str[i][0],displace=n_r[i]+0.5,angle=45,size=pyx.text.size.large))

      # Quarks
      x = n_x[i]
      y = n_y[i]
      for j,k in enumerate(q_color):
          ang = 180*j
          a = (ang/360.0)*2*3.14159
          qx = x + q_dis[i][j]*cos(a)
          qy = y + q_dis[i][j]*sin(a)
          q_pt = Point(qx,qy)
          if option>=0:
              #q.append(Circle(center=q_pt, radius=0.3, fill=[q_color[j]]).addLabel(r"\Pq",displace=0.01,size=pyx.text.size.large))
              print n_str[i][1][j]
              if j==0:
                  q.append(Circle(center=q_pt, radius=q_rad[i][j], fill=[q_color[j]]).addLabel(n_str[i][1][j],displace=0.01,size=pyx.text.size.large))
              if j==1:
                  q.append(Circle(center=q_pt, radius=q_rad[i][j], fill=[q_color[j]]))
                  aq.append(Circle(center=q_pt, radius=q_rad[i][j]/1.5, fill=[BLACK]).addLabel("\\textcolor{white}{"+n_str[i][1][j]+"}",displace=0.01,
                      size=pyx.text.size.large))
      

  outfilename = "%s_%d.pdf" % (outfilename,option)
  fd.draw(outfilename)

#################################################################

