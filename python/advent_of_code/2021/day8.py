import numpy as np
import sys

################################################################################

correct_digits = [[0, 'abcefg', 6], 
                  [1, 'cf', 2], 
                  [2, 'acdeg', 5], 
                  [3, 'acdfg', 5], 
                  [4, 'bdcf', 4], 
                  [5, 'abdfg', 5], 
                  [6, 'abdefg', 6], 
                  [7, 'acf', 3], 
                  [8, 'abcdefg', 7], 
                  [9, 'abcdfg', 6]
                  ]

correct_nsegments = []
for cg in correct_digits:
    correct_nsegments.append(cg[2])

print("Correct digits -----------")
print(correct_digits)
print()
print("Correct segments -----------")
print(correct_nsegments)
print()


################################################################################
def calc_nmatches(s1,s2):

    nmatches = 0
    for a in s1:
        if a in s2:
            nmatches += 1
    return nmatches

################################################################################
# Calc matches
matches = []
for cg in correct_digits:
    match = []
    for maybe_match in correct_digits:
        #if cg==maybe_match:
        #    continue
        nchars_matched = calc_nmatches(cg[1],maybe_match[1])
        match.append(nchars_matched)
    matches.append(match)


#####
for i,match in enumerate(matches):
    print(i,match)


#exit()

infilename = sys.argv[1]

data = np.loadtxt(infilename,delimiter='|',unpack=False,dtype=str)

################################################################################
print(data)
patterns = []
output = []
for d in data:
    p = d[0].split(' ')[:-1]
    o = d[1].split(' ')[1:]
    print(p)
    print(o)
    print(len(p))
    print(len(o))

    patterns.append(p)
    output.append(o)
################################################################################
print("Read in and processed the data!!!!!!!!!!! --------------")
print()

print("Part 1--------------------")
ndigits_of_interest = [2,3,4,7]

tot = 0
for o in output:
    #print(o)
    for val in o:
        ndigits = len(val)
        #print(ndigits)
        if ndigits in ndigits_of_interest:
            tot += 1

print(f"Total: {tot}")

################################################################################
print("Part 2        --------------------")
ndigits_of_interest = [2,3,4,7]

total_numbers = 0

for o,p in zip(output,patterns):
    print("--------------")
    newmap = {}
    newmap_flipped = {}
    for val in p:
        ndigits = len(val)
        #print(ndigits)
        # First find the 1,4,7,8
        if ndigits in ndigits_of_interest:
            if ndigits==2:
                newmap[val] = "1"
                newmap_flipped["1"] = val
            elif ndigits==3:
                newmap[val] = "7"
                newmap_flipped["7"] = val
            elif ndigits==4:
                newmap[val] = "4"
                newmap_flipped["4"] = val
            elif ndigits==7:
                newmap[val] = "8"
                newmap_flipped["8"] = val

        print(newmap)

    answer = ""
    nsolved = 0
    for number in o:
        print(f"Trying to figure out {number}")
        figured_it_out = False
        if len(number)==2:
            answer += "1"
            figured_it_out = True
        elif len(number)==3:
            answer += "7"
            figured_it_out = True
        elif len(number)==4:
            answer += "4"
            figured_it_out = True
        elif len(number)==7:
            answer += "8"
            figured_it_out = True
        elif len(number)==5:
            matched = np.array([True,True,True])
            possibilities = np.array([2,3,5])
            which_matched = []
            for key in newmap_flipped.keys():
                check_with = newmap_flipped[key]
                nmatches = calc_nmatches(number,check_with)
                for ip,poss in enumerate(possibilities):
                    idx1 = int(poss)
                    idx2 = int(key)
                    if nmatches != matches[idx1][idx2]:
                        matched[ip] = False
            good_number = possibilities[matched]
            print(len(good_number),good_number)
            if len(good_number==1):
                answer += str(good_number[0])
            figured_it_out = True
        elif len(number)==6:
            matched = np.array([True,True,True])
            possibilities = np.array([0,6,9])
            which_matched = []
            for key in newmap_flipped.keys():
                check_with = newmap_flipped[key]
                nmatches = calc_nmatches(number,check_with)
                for ip,poss in enumerate(possibilities):
                    idx1 = int(poss)
                    idx2 = int(key)
                    if nmatches != matches[idx1][idx2]:
                        matched[ip] = False
            good_number = possibilities[matched]
            print(len(good_number),good_number)
            if len(good_number==1):
                answer += str(good_number[0])
            figured_it_out = True
        else:
            for key in newmap:
                nmatches = calc_nmatches(number,key)
                digit = int(newmap[key])
                print(digit,nmatches)
        if figured_it_out:
            nsolved += 1
            print(f"We figured out number {number} and it is added to {answer}")
        else:
            answer += 'X'
            print(f"We couldn't figure out number {number} - {answer}")
    if nsolved==4:
        print(f"We solved {nsolved} characters!!!!")
        total_numbers += int(answer)
        print(f"MY ANSWER: {answer}")
    else:
        print("DISASTER!!!!!!!!!!!!!!!!!!")
        print(f"We solved {nsolved} characters!!!!")



print(f"Total numbers: {total_numbers}")

