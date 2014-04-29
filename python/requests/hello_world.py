import requests 
from bs4 import BeautifulSoup

# Grab the content from some website
r = requests.get('http://espn.go.com/mens-college-basketball/teams')

print r
print r.content


# Grab the content from some other website
r = requests.get('http://espn.go.com/mens-college-basketball/team/roster/_/id/399/albany-great-danes')

print r
print r.content


# Try to parse the webpage by looking for the tables.
soup = BeautifulSoup(r.content)

tables = soup.find_all('table')
print tables
