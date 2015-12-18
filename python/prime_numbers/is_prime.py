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


