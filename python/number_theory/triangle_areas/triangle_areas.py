import numpy as np
import matplotlib.pylab as plt

################################################################################
def area(a,b,c):
    s = 0.5*(a+b+c)
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))

    return A
################################################################################

lo = 3
hi = 31

x = []
y = []
for i in range(0,hi+1):
    x.append([])
    y.append([])

for L in range(lo,hi+1):

    maxA = int(np.floor(L/2))
    #print(L,maxA)

    minA = 1

    for a in range(maxA,minA-1,-1):
        for b in range(1,a+1):
            c = L-a-b
            #print(a,b,c)
            '''
            if c<b or c>a:
                break
            '''
            #print(L,a,b,c)
            #minA = c
            if not (a+b<=c or a+c<=b or b+c<=a) and c>0:
                A = area(a,b,c)
                print(L,a,b,c,A)
                #x.append(L)
                #x.append(min([a,b,c]))
                #y.append(A)
                x[L].append(min([a,b,c]))
                y[L].append(A)

    '''
    for a in range(1,L-1):
        for b in range(a,L-1):
            if a+b>=L:
                break
            c = L-a-b
            if a+b<=c or a+c<=b or b+c<=a:
                break
            #print(L,a,b,c)
            A = area(a,b,c)
            print(L,a,b,c,A)
            x.append(L)
            y.append(A)
    '''

plt.figure()
for L,(a,b) in enumerate(zip(x,y)):
    if len(a)>0:
        plt.plot(a,b,'-',label=str(L))
plt.legend()
plt.show()
