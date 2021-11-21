from bs4 import BeautifulSoup 

import requests

import sys
import pandas as pd

#url = "http://www.espn.com/nba/playbyplay?gameId=400974438"
#url = "siena-men-2019-2020.html"
# wget https://sienasaints.com/sports/mens-basketball/stats/2019-20 -O siena-men-2019-2020.html
# wget https://sienasaints.com/sports/womens-basketball/stats/2019-20 -O siena-women-2019-2020.html

# To download from url
#url = "https://sienasaints.com/sports/mens-basketball/stats/2019-20"
#url = "https://sienasaints.com/sports/womens-basketball/stats/2019-20"
#r  = requests.get(url)
#data = r.text

#data = open('siena-men-2019-2020.html','r')
#filename = 'siena-men-2019-2020.html'
filename = sys.argv[1]


#soup = BeautifulSoup(data,features='lxml')

'''
html = soup.prettify("utf-8")
with open("siena-men-2019-2020.html", "wb") as file:
    file.write(html)
'''
df = pd.read_html(filename)

tidx = 7
outfilename = filename.replace('.html','_BOTH.csv')
print(outfilename)

table = df[tidx]

table.to_csv(outfilename,index=False) # Don't write out the index



