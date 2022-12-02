import numpy as np
import sys

################################################################################

infilename = sys.argv[1]

fish = np.loadtxt(infilename,delimiter=',',unpack=False,dtype=int)

#ndays = 18
#ndays = 80
ndays = 256

'''
# Part 1
for n in range(ndays):
    print(f"After {n+1} days.....")
    print("fish")
    print(fish)

    nfish_new = len(fish[fish==0])

    fish_temp = fish 
    fish_temp[fish_temp==0] = 7
    fish_temp -= 1
    print("fish temp")
    print(fish_temp)

    nfish = len(fish_temp) 
    nfish_new = nfish + nfish_new

    fish_new = 8*np.ones(nfish_new,dtype=int)
    fish_new[0:nfish] = fish_temp

    fish = fish_new

    print(fish)
    print(f"# fish {len(fish)}")
'''

'''
tot = 0
for d in fish:
    tot += 1
    print(tot,d)
    for i in range(d,ndays):
        print(d,i,(i-d))
        if (i-d)%8==0:
            tot += (i-d)/8 + 1
print(tot)
'''


#'''
# Part 2
# Calculate for each number
numbers = np.zeros(9,dtype=int)
for i in range(1,6):
    numbers[i] = len(fish[fish==i])

print(fish)
print(numbers)

print("Starting the interations")
for n in range(ndays):
    print("-------------------")
    print(numbers)
    print(f"After day {n+1}")
    temp = numbers.copy()
    temp[0:-1] = numbers[1:]
    print('===')
    print(temp)
    print(numbers)
    print('===')
    temp[6] += numbers[0]
    print(temp)
    #if numbers[0] != 1:
    temp[8] = numbers[0]
    numbers = temp.copy()
    print(numbers)
print(numbers)
print(np.sum(numbers))
    



