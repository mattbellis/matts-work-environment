# Open a normal text file.
infile = open('input_data.dat')

for line in infile:
    print line
    vals = line.split()
    print vals

    # The vals are interpreted as strings, so you have to 
    # cast them as a float or int, if you want to use them
    # in an mathematical expression.
    print 2*float(vals[0])

# Open a normal csv file.
infile = open('input_data.csv')

for line in infile:
    print line
    vals = line.split(',')
    print vals

    # The vals are interpreted as strings, so you have to 
    # cast them as a float or int, if you want to use them
    # in an mathematical expression.
    print 2*float(vals[0])

