from bs4 import BeautifulSoup 

import requests


url = "http://www.espn.com/nba/playbyplay?gameId=400974438"
r  = requests.get(url)

data = r.text

soup = BeautifulSoup(data)

print(soup.prettify())

timestamp = []
player = []
score = []

for row in soup('tr', attrs={'class':'scoring-play'}):

    print(row)
    print(" ---------------------")
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]

    timestamp.append(cols[0])

    play_info = cols[2]
    playername = "%s %s" % (play_info.split()[0],play_info.split()[1])
    player.append(playername)

    score.append(cols[3])


for t,p,s in zip(timestamp, player, score):
    print("{0:5s} {1:20s} {2:10s}".format(t,p,s))

