from bs4 import BeautifulSoup 

import requests


#url = "http://www.espn.com/nba/playbyplay?gameId=400974438"
url = "https://sienasaints.com/sports/mens-basketball/stats/2019-20"
r  = requests.get(url)

data = r.text

soup = BeautifulSoup(data,features='lxml')

print("Got the soup!")

#print(soup.prettify())

timestamp = []
player = []
score = []

data = {}

#for row in soup('tr', attrs={'class':'Game By Game - Team Statistics'}):
#for row in soup('caption', attrs={'text':'Game By Game - Team Statistics'}):
for row in soup('table', attrs={'class':"sidearm-table highlight-column-hover"}):

    print("------")
    #print(row)
    print("------")

    tr = row.find_all('tr')
    #print(cols)
    print('=========================')
    #print(cols)
    for t in tr:
        print("-------")
        print(t)
        tds = t.find_all('td')
        for td in tds:
            print('==========')
            print(td)


    '''
    if row is None:
        continue

    if row.find('scope')>=0 and row.find('col')>=0:
        field = row.split('>')[1].split('<')[0]
        if field not in list(data.keys()):
            data[field] = []
    '''
    '''
    print(" ---------------------")
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]

    timestamp.append(cols[0])

    play_info = cols[2]
    playername = "%s %s" % (play_info.split()[0],play_info.split()[1])
    player.append(playername)

    score.append(cols[3])
    '''

print(data)

'''
for t,p,s in zip(timestamp, player, score):
    print("{0:5s} {1:20s} {2:10s}".format(t,p,s))
'''
