import numpy as np
import numpy.random as rnd

ntrials = 100000
nteams = 12

mean = 100
std = 20

ngames = 6
badluck = 0

for i in range(0,ntrials):

    tot = np.zeros(nteams)

    winloss = np.zeros(nteams,dtype='int')

    for j in xrange(ngames):

        teams = np.arange(0,12)

        scores = rnd.normal(mean,std,nteams)
        #print scores

        tot += scores

        # Shuffle the team order. Games consist of consecutive pairs (0-1, 2-3, ...)
        rnd.shuffle(teams)
        #print teams
        for k in range(0,nteams,2):
            # Get the two teams
            t0 = teams[k]
            t1 = teams[k+1]

            #print t0,t1,scores[t0],scores[t1]

            if scores[t0]>scores[t1]:
                winloss[t0]+=1
            else:
                winloss[t1]+=1

    #print tot
    #print winloss

    mostpoints = max(tot)
    #print tot.tolist()
    index = tot.tolist().index(mostpoints)
    #print index
    if winloss[index]<=1:
        badluck += 1

print badluck
print float(badluck)/ntrials

    
