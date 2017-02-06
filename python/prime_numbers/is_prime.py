def is_prime(n):
    #for i in range(2,int(n/2)+2):
    i = 2
    while i < int(n/i)+2:
        if n%i==0:
            #print('not prime')
            return False # is not prime
        i+= 1
    return True # is prime


def is_square(n):
    from math import sqrt 
    a = int(sqrt(n))
    if a*a == n:
        return True # is a perfect square
    else:
        return False


def is_odd(n):
    if n%2 == 0:
        return False # is odd
    else:
        return True


def get_factors(n):
    factors = []
    i = 2
    max_val = int(n/i)+2
    while i < max_val:
        if n%i==0:
            #print('not prime')
            factors.append(i)
            max_val = n/i
            factors.append(max_val)
        i+= 1

    factors.append(n)
    return factors

def relatively_prime(a,b):
    factorsa = get_factors(a)
    factorsb = get_factors(b)

    for i in factorsa:
        if i in factorsb:
            return False

    return True
