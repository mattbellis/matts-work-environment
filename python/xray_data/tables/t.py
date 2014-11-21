f = open('tmp2.tmp')
for line in f:
    #print len(line)
    if not (line[0] == '1' and len(line)==1):
        print line

