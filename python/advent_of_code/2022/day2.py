import sys
import numpy as np

# Part A
'''
infilename = sys.argv[1]

def who_wins(a,b):
    points = -1
    if (a=="A" and b=="Y") or  \
       (a=="B" and b=="Z") or  \
       (a=="C" and b=="X"):
       points = 6
    elif (a=="A" and b=="X") or  \
       (a=="B" and b=="Y") or  \
       (a=="C" and b=="Z"):
       points = 3
    else:
       points = 0

    return points 

obj_points = {"X":1, "Y":2, "Z":3}

# Part A
opp,me = np.loadtxt(infilename,dtype=str,unpack=True)

print(opp)
print(me)

total_points = 0
for a,b in zip(opp,me):
    x = who_wins(a,b)
    y = obj_points[b]
    print(x,y)
    total_points += (x + y)

print(f"total_points: {total_points}")
'''

# Part B
infilename = sys.argv[1]

def which_do_I_need(a,b):
    my_play = "M"
    # x=rock
    # y=paper
    # z=scissors
    if (b=="X"): # I lose
        if (a=="A"):
            my_play = "z"
        elif (a=="B"):
            my_play = "x"
        elif (a=="C"):
            my_play = "y"
    elif (b=="Y"): # We draw
        if (a=="A"):
            my_play = "x"
        elif (a=="B"):
            my_play = "y"
        elif (a=="C"):
            my_play = "z"
    elif (b=="Z"): # I win
        if (a=="A"):
            my_play = "y"
        elif (a=="B"):
            my_play = "z"
        elif (a=="C"):
            my_play = "x"

    return my_play 

obj_points = {"x":1, "y":2, "z":3}
win_lose_draw_points = {"X":0, "Y":3, "Z":6}

# Part A
opp,me = np.loadtxt(infilename,dtype=str,unpack=True)

print(opp)
print(me)

total_points = 0
for a,b in zip(opp,me):
    x = win_lose_draw_points[b]
    my_play = which_do_I_need(a,b)
    y = obj_points[my_play]
    print(x,y)
    total_points += (x + y)

print(f"total_points: {total_points}")
