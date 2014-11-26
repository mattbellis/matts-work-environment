import numpy as np

g = -9.8 # m/s
vmag = 60. # m/s
angle = 40. # degrees
height = 100.0 # meters
vx = np.array([])
vy = np.array([])
ttot = np.array([])

angle = np.deg2rad(angle)

def part0(vmag,angle):
    vx = vmag*np.cos(angle)
    return vx

def part1(vmag,angle):
    vy = vmag*np.sin(angle)
    return vy

def part2(vy):
    t0 = -vy/g
    return t0


answers = [40.0,38.6,3.9]
alt_answers = []

vx = part0(vmag,angle)
alt_answers.append(vx)
print vx

vy = part1(vmag,angle)
alt_answers.append(vy)
print vy

t0 = part2(vy)
alt_answers.append(t0)
print t0

print answers
print alt_answers

for a,alt in zip(answers,alt_answers):

    if np.abs((a-alt)/alt)<0.05:
        print "Correct!!! %f %f" % (a,alt)
    else:
        print "Incorrect!!! %f %f" % (a,alt)
