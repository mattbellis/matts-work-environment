import sys

infile0 = open(sys.argv[1])
infile1 = open(sys.argv[2])

for line0,line1 in zip(infile0,infile1):
    if line0 != line1:
        print("====================")

        output0 = ' '.join(line0.split(','))
        output1 = ' '.join(line1.split(','))
        #print(output0.strip())
        #print(output1.strip())

        for o in [line0,line1]:
            vals = o.split(',')
            output = ""
            for v in vals:
                output += "{0:5} ".format(v)
            print(output.strip())
