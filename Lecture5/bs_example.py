#import packages for scraping

from bs4 import BeautifulSoup
from urllib.request import urlopen
from datetime import datetime

# Construct url
base_url = "https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date"
q_date = datetime.today().strftime('%Y%m%d')
print(q_date)

url = base_url+q_date
print(url)

# Get webpage source with Soup
page = urlopen(url)
soup = BeautifulSoup(page, "html.parser")

# Get data from HTML source
# Signature: find_all(name, attrs, recursive, string, limit, **kwargs)
# The find_all() method looks through a tag’s descendants and retrieves
# all descendants that match your filters. I gave several examples in
# Kinds of filters, but here are a few more:

movie_titles = soup.find_all('div', 'tit5')
movie_points = soup.find_all('td', 'point')
type(movie_titles)
#print(movie_titles)
#print(movie_points)

for title in movie_titles:
    #print(title)
    print(title.a.string)
    print(title.a.attrs['href'])

# movie_dict = {}
# for (link, point) in zip(movie_titles, movie_points):
#     print(link.a.string, point.string)
#     movie_dict[link.a.string] = point.string

# print(movie_dict)


