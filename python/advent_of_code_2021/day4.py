import numpy as np
import sys

################################################################################
def look_for_number_in_board(board,number):
    for i,row in enumerate(board):
        r = row.tolist()
        if number in r:
            return i,r.index(number)
    return -999,-999
################################################################################
################################################################################
def check_truth_board(board):
    for i,row in enumerate(board):
        if len(row[row])==0:
            return True
    for i,col in enumerate(board.T):
        if len(col[col])==0:
            return True
    return False
################################################################################
################################################################################
def calc_winner(board,truth,n):
    print(board)
    print(truth)
    print(board[truth])
    print(np.sum(board[truth]))
    print(n)
    print(np.sum(board[truth])*n)
    return np.sum(board[truth])*n
################################################################################

infilename = sys.argv[1]
infile = open(infilename)

numbers = None
boards = []
boards_truth = []
board = []
new_board = True

for i,line in enumerate(infile):
    #print(i,line)
    if i==0:
        numbers = np.array(line.split(',')).astype(int)
    else:
        if len(line.strip()) == 0:
            #print("HERE!!!!!")
            if len(board)>0:
                board = np.array(board)
                boards.append(board)
            board = []
            new_board = True
            continue
        elif new_board==True and line != "":
            #print(np.array(line.strip().split()))
            row = np.array(line.strip().split()).astype(int)
            board.append(row)

# Need to do this at the end
board = np.array(board)
boards.append(board)

################################################################################
print(numbers)
for board in boards:
    tb = np.ones(shape=board.shape,dtype=bool)
    print()
    print(tb)
    boards_truth.append(tb)
    print()
    for row in board:
        print(board)
################################################################################

print("CHcking the numbers!!!!")
nboards = len(boards)
print(nboards)
winning_boards = np.zeros(nboards,dtype=bool)
#exit()

for n in numbers:
    print(n)
    for boardcount,(truth,board) in enumerate(zip(boards_truth,boards)):
        i,j = look_for_number_in_board(board,n)
        print(i,j)
        if i<0:
            continue

        # If it won, then skip
        if winning_boards[boardcount]:
            continue

        truth[i][j] = False
        x = check_truth_board(truth)
        if x is True:
            winning_boards[boardcount] = True
            print("WINTER =============================")
            calc_winner(board,truth,n)
            #exit()
            #break

        print("Winning!!!!: ",len(winning_boards[winning_boards]))
        if len(winning_boards[winning_boards])==nboards:
            exit()




