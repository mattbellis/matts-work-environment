import sys

n = int(sys.argv[1])

nmax = (n+2)*(n+1)*n/6

print "max:  %d" % (nmax)
output = ""
for x in range(0,n):
    for y in range(0,x+1):
        for z in range(0,y+1):

            nbin = (z)*(3*n*(n+1)-(3*n+2)*(z+1) + (z+1)*(z+1))/6 + (y)*(2*n-(y+1))/2 + x

            print "nmax: %d x,y,z: %d %d %d \tnbin: %d" % (nmax,x,y,z,nbin)

            output += "%d " % (nbin)
        output += "\n"
    output += "\n"
print output

