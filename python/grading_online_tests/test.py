import numpy as np

################################################################################
def yea_or_nay(n0,n1,tol=0.05):

    if np.abs((n0-n1)/n0)<0.05:
        print "\tCorrect!!! %f %f" % (n0,n1)
        return 1
    else:
        print "\tIncorrect!!! %f %f" % (n0,n1)
        return 0

################################################################################

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


# Correct
# [45.962666587138678, 38.567256581192353, 3.9354343450196274]

#student_answers = [46.0,38.6,3.9]
student_answers = [46.0,20.6,2.1]

solutions = []
alt_solutions = []

vx = part0(vmag,angle)
solutions.append(vx)
alt_solutions.append(vx)
#print vx

vy = part1(vmag,angle)
solutions.append(vy)
alt_solutions.append(vy)
#print vy

t0 = part2(vy)
solutions.append(t0)
alt_solutions.append(part2(student_answers[1]))
#print t0

print student_answers
print solutions
print alt_solutions

for sa,sols,alt_sols in zip(student_answers,solutions,alt_solutions):

    print "True answer........"
    org_sol = yea_or_nay(sa,sols)
    if not org_sol and sols is not alt_sols:
        print "Partial credit....."
        partial_sol = yea_or_nay(sa,alt_sols)
