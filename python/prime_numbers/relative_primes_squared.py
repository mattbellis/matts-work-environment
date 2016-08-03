from is_prime import is_prime,is_square,is_odd,relatively_prime,get_factors

import sys

max_val = 10000

odds = []
odds1 = []
odds3 = []
odds5 = []
odds7 = []
odds9 = []

primes = []

# Get all the odds and the
# ones that end in 1,3,5,7,9
for i in xrange(max_val):
    if is_prime(i):
        primes.append(i)

    if is_odd(i):
        odds.append(i)

        remainder = i%10
        if remainder==1:
            if i!=1:
                odds1.append(i)
        elif remainder==3:
            odds3.append(i)
        elif remainder==5:
            odds5.append(i)
        elif remainder==7:
            odds7.append(i)
        elif remainder==9:
            odds9.append(i)

# OK, now check the combinations
ncombs = 0
nprime = 0
for a in odds1:
    for b in odds9:
        #if relatively_prime(a,b):
        if is_prime(a) and is_prime(b):
            ncombs += 1
            num = (a*a + b*b)/2
            if is_prime(num)==False:
                0#print get_factors(num)
            else:
                print a,b,a*a,b*b,a*a+b*b,num,is_prime(num)
                nprime += 1

print
print ncombs
print nprime
print float(nprime)/ncombs
print
print len(primes)/float(max_val)




