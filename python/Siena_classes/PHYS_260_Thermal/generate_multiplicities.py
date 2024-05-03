import numpy as np
import matplotlib.pylab as plt

from scipy.special import factorial

qtot = 90
N_A = 40
N_B = 20

mult_A = []
mult_B = []

for q_A in range(0,qtot+1):

    q_B = qtot - q_A

    mult_A.append(factorial(q_A+N_A)/(factorial(q_A)*factorial(N_A)))
    mult_B.append(factorial(q_B+N_B)/(factorial(q_B)*factorial(N_B)))

    #print "%-10d & %-10d & %-10d & %-10d \\\\" % (q_A,mult_A[q_A],q_B,mult_B[q_A])
    print("%-10d & %-10d & %-10d & %-10d & %-10d \\\\" % (q_A,mult_A[q_A],q_B,mult_B[q_A],mult_A[q_A]*mult_B[q_A]))


for i in range(0,len(mult_A)-1):

    if i > 0:
        print(i,np.log(mult_A[i+1]) - np.log(mult_A[i-1]) , qtot-i,np.log(mult_B[i-1]) - np.log(mult_B[i+1]))

# Plot this
q = range(0,qtot+1)

q = np.array(q)
print("hree")
print(q)
epsilon = 0.1 

k = 8.6e-5 # eV/K

S_A = k*np.log(mult_A)
S_B = k*np.log(mult_B)

S_tot = S_A+S_B

plt.figure(figsize=(8,5))
plt.plot(q*epsilon,S_A,'--',linewidth=3,label='Entropy of Solid A')
plt.plot(q*epsilon,S_B,'-.',linewidth=3,label='Entropy of Solid B')
plt.plot(q*epsilon,S_tot,'-',linewidth=3,label='Total entropy')
#plt.xlabel(r'$q_A\times \epsilon$ (eV)',fontsize=24)
plt.xlabel(r'$E_A$ (eV)',fontsize=24)
plt.ylabel(r'$S$ (eV/K)',fontsize=24)
plt.grid(which='both')
plt.legend(loc='center right')

plt.tight_layout()

plt.savefig('two_solids_entropy_vs_E.png')



plt.show()
