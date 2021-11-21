from bs4 import BeautifulSoup 

import requests

#url = "http://www.espn.com/nba/playbyplay?gameId=400974438"
#url = "siena-men-2019-2020.html"
# wget https://sienasaints.com/sports/mens-basketball/stats/2019-20 -O siena-men-2019-2020.html

# To download from url
#url = "https://sienasaints.com/sports/mens-basketball/stats/2019-20"
#r  = requests.get(url)
#data = r.text

data = open('siena-men-2019-2020.html','r')

soup = BeautifulSoup(data,features='lxml')

'''
html = soup.prettify("utf-8")
with open("siena-men-2019-2020.html", "wb") as file:
    file.write(html)
'''

print("Got the soup!")

#print(soup.prettify())

timestamp = []
player = []
score = []

data = {}

#for table in soup('table', attrs={'class':"sidearm-table highlight-column-hover"}):
tables = soup.body.find_all('table')
for table in tables:
    caption = table.find_all('caption')
    print(caption)

    if caption[0].text.find('Comparison')>=0:
        #print(table)
        header = ""
        trs = table.find_all('tr')
        for tr in trs:
            print("-------")
            #print(tr)
            tds = tr.find_all('td')
            output = ""
            for td in tds:
                print("---")
                print(td)
                a = td.find_all('a')
                print(a)
                if len(a)==1:
                    if 'aria-label' in a[0].attrs:
                        print("HERER!!!!!!!!!!!!")
                        print(a[0].attrs)
                        text = a[0].text.strip()
                        print(a[0]['aria-label'])
                        gamedate = a[0]['aria-label'].strip().split()[-1]
                        output += f"{text},{gamedate}"
                if 'data-label' in td.attrs:
                    print(td['data-label'],td.text)
                    text = td.text.strip()
                    #print(type(text))
                    #text = text,replace('\t','')
                    output += f",{text}"
            output += "\n"
            print(output)
    

exit()

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
            print('td ==========')
            print(td)
            if td != None:
                #print(td.getAttribute('data-label'))
                print(td.string)
                print(td.attributes)
            print('td END ==========')


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
