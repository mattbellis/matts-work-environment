def is_prime(n):
    for i in range(2,int(n/2)+2):
        if n%i==0:
            #print('not prime')
            return False # is not prime
    return True # is prime


