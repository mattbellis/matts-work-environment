import sys
import numpy as np

from anytree import Node, RenderTree, AsciiStyle, Walker, PreOrderIter

import copy

infilename = sys.argv[1]
infile = open(infilename,'r')

monkeys = []
monkey_num = -1

# Part A
for line in infile:
    if line.find("Monkey")>=0:
        monkey_num = int(line.split()[1].split(':')[0])
        monkey = {monkey_num: {}}
        monkey['ninspections'] = 0
        monkeys.append(copy.deepcopy(monkey))
    elif line.find("Starting")>=0:
        items = np.array(line.split(':')[1].split(',')).astype(int).tolist()
        monkeys[monkey_num]['items'] = items
    elif line.find("Operation")>=0:
        items = line.split()[-2:]
        monkeys[monkey_num]['operation'] = items
    elif line.find("divisible")>=0:
        items = line.split()[-1]
        monkeys[monkey_num]['divisibleby'] = int(items)
    elif line.find("true")>=0:
        items = line.split()[-1]
        monkeys[monkey_num]['true'] = int(items)
    elif line.find("false")>=0:
        items = line.split()[-1]
        monkeys[monkey_num]['false'] = int(items)


#print(monkeys)
for monkey in monkeys:
    print(monkey)
print()

# Process
# Part A
'''
nrounds = 20
nmonkeys = len(monkeys)
for i in range(nrounds):
    for imon in range(nmonkeys):
        monkey = monkeys[imon]
        nitems = len(monkey['items'])
        for n in range(nitems):
            # Do operation
            item = monkey['items'].pop()
            print(f"{imon}  INSPECTING {item}")
            operation = monkey['operation']
            if operation[-1] == 'old':
                item *= item
            elif operation[0] == '+':
                item += int(operation[1])
            elif operation[0] == '*':
                item *= int(operation[1])

            item = int(np.floor(item/3))

            test = item%monkey['divisibleby']==0

            idx = -1
            if test is True:
                idx = monkey['true']
            else:
                idx = monkey['false']

            monkeys[idx]['items'].insert(0,item)

            monkey['ninspections'] += 1


print()
#print(monkeys)
for monkey in monkeys:
    print(monkey)




nrounds = 20
nmonkeys = len(monkeys)
for i in range(nrounds):
    for imon in range(nmonkeys):
        monkey = monkeys[imon]
        nitems = len(monkey['items'])
        for n in range(nitems):
            # Do operation
            item = monkey['items'].pop()
            print(f"{imon}  INSPECTING {item}")
            operation = monkey['operation']
            if operation[-1] == 'old':
                item *= item
            elif operation[0] == '+':
                item += int(operation[1])
            elif operation[0] == '*':
                item *= int(operation[1])

            item = int(np.floor(item/3))

            test = item%monkey['divisibleby']==0

            idx = -1
            if test is True:
                idx = monkey['true']
            else:
                idx = monkey['false']

            monkeys[idx]['items'].insert(0,item)

            monkey['ninspections'] += 1


print()
#print(monkeys)
for monkey in monkeys:
    print(monkey)
'''

# Part B
nrounds = 20
nmonkeys = len(monkeys)
for i in range(nrounds):
    if i%10==0:
        print(i)
        for monkey in monkeys:
            print(monkey)
    for imon in range(nmonkeys):
        monkey = monkeys[imon]
        nitems = len(monkey['items'])
        for n in range(nitems):
            # Do operation
            item = monkey['items'].pop()
            #print(f"{imon}  INSPECTING {item}")
            operation = monkey['operation']
            if operation[-1] == 'old':
                item *= item
            elif operation[0] == '+':
                item += int(operation[1])
            elif operation[0] == '*':
                item *= int(operation[1])

            #item = int(np.floor(item/3))

            test = item%monkey['divisibleby']==0

            idx = -1
            if test is True:
                idx = monkey['true']
                #print("TRUEEEEEEEEEEEEEEEEEEEEEE")
            else:
                idx = monkey['false']

            monkeys[idx]['items'].insert(0,item)

            monkey['ninspections'] += 1


print()
#print(monkeys)
for monkey in monkeys:
    print(monkey)




