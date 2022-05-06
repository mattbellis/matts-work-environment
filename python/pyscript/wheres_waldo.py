import numpy as np

################################################################################
def print_grid(grid):
    output = ""
    for i in range(nrows):
        for j in range(ncols):
            output += f"{grid[i][j]} "
        output += "\n"

    print(output)
################################################################################

nrows = 20
ncols = 20

grid = np.ones((20,20),dtype=str)

for i in range(nrows):
    for j in range(ncols):
        rnd = np.random.randint(65,91)
        grid[i][j] = chr(rnd)
print(grid)



#########################################
# Which way?
rnd = np.random.randint(0,3)
rndrow = np.random.randint(0,nrows-1)
rndcol = np.random.randint(0,ncols-1)

print(rnd,rndrow,rndcol)

waldo = ['W','A','L','D','O']
num = len(waldo)

if rnd == 0: # left to right
    if rndcol>ncols-num:
        rndcol = ncols-num
    for i in range(num):
        grid[rndrow][rndcol+i] = waldo[i]
elif rnd == 1: # right to left 
    for i in range(num):
        if rndcol<num-1:
            rndcol = num-1
        grid[rndrow][rndcol-i] = waldo[i]




print_grid(grid)



