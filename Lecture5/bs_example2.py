
from bs4 import BeautifulSoup
from urllib.request import urlopen

base_url = "https://www.kmu.ac.kr/uni/main/page.jsp?mnu_uid=143"

page = urlopen(base_url)
soup = BeautifulSoup(page, "html.parser")

#title
#date
#hit number

titles = soup.find_all('td','subject')
for title in titles:
    #print(title)
    print(title.a.string)

dates = soup.find_all('td','date')
for date in dates:
    #print(date)
    print(date.string)

hits = soup.find_all('td','hit')
for hit in hits:
    print(hit.string)

notice = []
for title, date, hit in zip(titles, dates, hits) :
    print(title.a.string)
    print(date.string)
    print(hit.string)

    notice.append([title.a.string,date.string,hit.string])

print(notice)

