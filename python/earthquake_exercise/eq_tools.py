import numpy as np
import matplotlib.pylab as plt

def draw_earth(radii):
#colors = ['y','b','r','g']
    for i,r in enumerate(radii):
        circle = plt.Circle((0, 0), radius=r,alpha=0.05)#,fc=colors[i])
        plt.gca().add_patch(circle)
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)

def normal(r,x,y):
    return r

def snells(theta0,v0,v1):
    sin_theta1 = np.sin(theta0)*(v1/v0)
    theta1 = np.arcsin(sin_theta1)
    return theta1

def mag(x,y):
    r = np.sqrt(x*x + y*y)
    return r

def rel_angle(x0,y0, x1,y1):
    mag0 = mag(x0,y0)
    mag1 = mag(x1,y1)
    
    cos_theta = (x0*x1 + y0*y1)/(mag0*mag1)
    
    theta = np.arccos(cos_theta)
    return theta

def linear(m,b,x):
    y = b + m*x
    return y

def radial_dist(x,y):
    r2 = x*x + y*y
    r = np.sqrt(r2)
    return r

def vector_representation(x0,y0, x1,y1, norm=False):
    #theta = np.arctan2(y1-y0,x1-x0)
    dx = x1-x0
    dy = y1-y0
    vec_mag = 1.0
    if norm==True:
        vec_mag = mag(dx,dy)
    x = dx/vec_mag
    y = dy/vec_mag
    #y = vec_mag*np.sin(theta)
    return x,y

def new_y(x0,y0,x1,theta):
    y = (x1-x0)*np.tan(theta) + y0
    return y

def sgn(x):
    if x<0:
        return -1
    else:
        return +1 

def intersection(x1,y1, x2,y2, r):
    dx = x2-x1
    dy = y2-y1
    dr = mag(dx,dy)
    D = x1*y2 - x2*y1
    
    pts = []
    radical = r**2 * dr**2 - D**2 
    if radical<0:
        return None
    else:
        x = (D*dy + sgn(dy)*dx*np.sqrt(radical))/dr**2
        y = (-D*dx + np.abs(dy)*np.sqrt(radical))/dr**2
        pts.append(np.array([x,y]))
        if radical==0:
            return pts
        else:
            x = (D*dy + sgn(dy)*dx*-1*np.sqrt(radical))/dr**2
            y = (-D*dx + np.abs(dy)*-1*np.sqrt(radical))/dr**2
            pts.append(np.array([x,y]))
            return pts
        
def radial_pts(x,y,radius=1.0):
    theta = rel_angle(1.0, 0.0, x, y)
    rx = radius*np.cos(theta)
    ry = radius*np.sin(theta)
    
    rx *= x/np.abs(x)
    ry *= y/np.abs(y)
    
    return rx,ry

def trace_to_radius(x0,y0,angle,current_radius,new_radius,current_vel,new_vel):
    # x0 and y0 are the points where the ray starts
    # angle is the angle of that ray
    # radius is the radius that we are looking to see if it intercepts

    # return the point at which it intersects the new circle and 
    # the angle at which it enters or exits that radius
    
    #print "input: ",x0,y0,angle,current_radius,new_radius
    # Extend the line
    x = 1.0
    y = new_y(x0,y0,x,angle)
    #print x,y

    # See if it intersects our radius
    pts = intersection(x0,y0,x,y,new_radius)
    #print pts

    closest = None
    if pts is None:
        return None, None, angle
    
    elif pts is not None:
        #print "intersection pts: ",pts
        closest = None
        if len(pts)==1:
            closest = pts[0]
            return closest[0],closest[1],angle
        if len(pts)>1:
            d0 = mag(x0-pts[0][0],y0-pts[0][1])
            d1 = mag(x0-pts[1][0],y0-pts[1][1])
            if d0<d1:
                if new_radius==current_radius:
                    closest = pts[1]
                else:
                    #print "THIS CLOSEST"
                    closest = pts[0]

            else:
                if new_radius==current_radius:
                    closest = pts[0]
                else:
                    closest = pts[1]
                    
    #print "closest: ",closest
    ray = [[x0,closest[0]],[y0,closest[1]]]
    rd0 = radial_pts(closest[0],closest[1])
    
    # Next layer
    #print "TWO"
    vx,vy = vector_representation(ray[0][1],ray[1][1],x0,y0)
    cx = ray[0][1] # Circle x
    cy = ray[1][1] # Circle y
    t0 = rel_angle(cx,cy,vx,vy)

    t1 = snells(t0,current_vel,new_vel)
    norm_angle = rel_angle(cx,cy,1.0, 0.0)
    #print "norm: ",np.rad2deg(norm_angle)
    #print "t0: ",np.rad2deg(t0)
    #print "t1: ",np.rad2deg(t1)
    if new_radius<current_radius:
        #angle -= (t1-t0) # Change in angle is the difference between t0 and t1
        angle = np.pi - (np.pi - norm_angle - t1)
        #print "HERE!"
    else:
        radial_angle = rel_angle(1.0, 0.0, cx,cy)
        if cx>0 and cy<0:
            radial_angle = -np.abs(radial_angle)
        #print "radial angle: ",np.rad2deg(radial_angle)
        angle = radial_angle - np.abs(t1)
        '''
        if cx>0 and cy<0:
            angle = radial_angle - t1
        else:
            angle = radial_angle + t1
        '''
        #print "there"
    #angle = norm_angle - t1
    #print "new angle: ",np.rad2deg(angle)
    
    return closest[0],closest[1],angle
