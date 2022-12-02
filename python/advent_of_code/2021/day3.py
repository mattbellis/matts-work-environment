import numpy as np
import sys

testdata = ['00100',
'11110',
'10110',
'10111',
'10101',
'01111',
'00111',
'11100',
'10000',
'11001',
'00010',
'01010']

testdata = np.array(testdata)


infilename = sys.argv[1]
data = np.loadtxt(infilename,unpack=False,dtype=str)

#print(data)

def func(data):

    temp_data = []
    for d in data:
        #print(d)
        temp_data.append([char for char in d])
        #print("-------")
        #print(d)
        #print(d.split())
        #temp_data.append(np.array(d.split()))
    data = np.array(temp_data).astype(int).T
    print(data)
    #exit()

    nbits = len(data)
    gamma = 0
    epsilon = 0
    icount = 0
    #print("nbits: ",nbits)
    for nbit in range(nbits-1,-1,-1):
    #for nbit in range(0,nbits):
        #print("----------")
        bit = data[nbit]
        #print(icount,2**icount,nbit,bit)
        n1 = len(bit[bit==1])
        n0 = len(bit[bit==0])
        if n1>n0:
            gamma +=   (2**icount)*1
            epsilon += (2**icount)*0
        else:
            gamma +=   (2**icount)*0
            epsilon += (2**icount)*1
        #print(n0,n1,gamma,epsilon)
        icount += 1


    print(gamma, epsilon, gamma*epsilon)

func(testdata)

func(data)

################################################################################
# Part 2

print("Part 2 ---------------")
print()

def find_most_common_bit(data,nbit):

    #print("In find_most_common_bit!")

    temp_data = []
    for d in data:
        temp_data.append([char for char in d])
    temp_data = np.array(temp_data).astype(int).T
    #print(temp_data)

    bit = temp_data[nbit]
    #print(bit)
    n1 = len(bit[bit==1])
    n0 = len(bit[bit==0])

    if n1>n0:
        return 1,0,False
    elif n0>n1:
        return 0,1,False
    elif n0==n1:
        return 0,0,True


def func2(data):

    print("Start!!!! ========================================== ")
    print("nentries: ",len(data))
    nbits = len(data[0])
    #print("nbits: ",nbits)

    keep_most = np.ones(len(data)).astype(bool)
    remaining_data_most = data[keep_most]

    keep_least = np.ones(len(data)).astype(bool)
    remaining_data_least = data[keep_least]

    for nbit in range(0,nbits):
        #keep_most = np.ones(len(remaining_data_most)).astype(bool)
        #remaining_data_most = remaining_data_most[keep_most]
        #print("remaining data --- ", nbit)
        #print(remaining_data_most)
        #print("----------")
        #bit = temp_data[nbit]
        #print(icount,2**icount,nbit,bit)
        most_and_least = find_most_common_bit(remaining_data_most,nbit)
        mcb_most = str(most_and_least[0])
        lcb_most = str(most_and_least[1])
        issame_most = most_and_least[2]
        print("most common bit:  ",mcb_most)
        print("least common bit: ",lcb_most)
        print("is same?:         ",issame_most)

        most_and_least = find_most_common_bit(remaining_data_least,nbit)
        mcb_least = str(most_and_least[0])
        lcb_least = str(most_and_least[1])
        issame_least = most_and_least[2]
        print("most common bit:  ",mcb_least)
        print("least common bit: ",lcb_least)
        print("is same?:         ",issame_least)

        print("LEN! :",len(remaining_data_most))
        if len(remaining_data_most)>1:
            for i,d in enumerate(remaining_data_most):
                if issame_most:
                    mcb_most = "1"
                if d[nbit] != mcb_most:
                    keep_most[i] = False

        print("LEN! :",len(remaining_data_least))
        if len(remaining_data_least)>1:
            for i,d in enumerate(remaining_data_least):
                if issame_least:
                    lcb_least = "0"
                if d[nbit] != lcb_least:
                    keep_least[i] = False

        #print("here")
        #print(keep_most)
        if len(remaining_data_most)>1:
            temp_remaining_data_most = remaining_data_most[keep_most]
            keep_most = np.ones(len(remaining_data_most[keep_most])).astype(bool)
            remaining_data_most = temp_remaining_data_most

        if len(remaining_data_least)>1:
            temp_remaining_data_least = remaining_data_least[keep_least]
            keep_least = np.ones(len(remaining_data_least[keep_least])).astype(bool)
            remaining_data_least = temp_remaining_data_least

    print("Surviving ----------------------------")
    print("Most!")
    print(keep_most)
    print(remaining_data_most[keep_most])

    print("Least!")
    print(keep_least)
    print(remaining_data_least[keep_least])

    def string_to_binary(value):
        print("In string to binary!")
        nbits = len(value)
        icount = 0
        number = 0
        print(value)
        for i in range(nbits-1,-1,-1):
            n = int(value[i])
            print(icount,n,2**icount)
            number += n*(2**icount)
            icount += 1

        return number

    print("These!!!!! ---------------- ")
    print(remaining_data_most[keep_most])
    print(remaining_data_least[keep_least])
    #print(string_to_binary(remaining_data_most[keep_most]))
    #print(string_to_binary(remaining_data_least[keep_least]))

    o2 = string_to_binary(remaining_data_most[keep_most][0])
    co2 = string_to_binary(remaining_data_least[keep_least][0])

    print(o2, co2, o2*co2)

for t in testdata:
    print(t)
print()
func2(testdata)

func2(data)

################################################################################
# Part 2

