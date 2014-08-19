import numpy as np

message = "The idea behind digital computers may be explained by saying that these machines are intended to carry out any operations which could be done by a human computer. Alan Turing. Also Soundgarden is awesome."

totstring = ""
for m in message:
    #print m.upper()
    if m==".":
        mystr = "99"
    elif m==" ":
        mystr = "00"
    else:
        mystr = "%02d" % (ord(m.upper())-64)

    print mystr
    totstring += mystr

print totstring

