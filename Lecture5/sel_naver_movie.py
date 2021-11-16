# 팩키지 임포트
from selenium import webdriver
from datetime import datetime

#  driver라는 이름의 webdriver 객체를 만들어 주자.
#  Chrome의 경우 | 아까 받은 chromedriver의 위치를 지정해준다.

driver = webdriver.Chrome('/Users/baeseongho/webdriver/chromedriver')

#driver = webdriver.Chrome(driver_path)
#driver.implicitly_wait(3)

base_url = "https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date="
q_date = datetime.today().strftime('%Y%m%d')
print(q_date)
url = base_url+q_date
print(url)

# url에 접근한다.
driver.get(url)
titles = driver.find_elements_by_class_name('tit5')
points = driver.find_elements_by_class_name('point')

for title in titles:
    print(title.text)

for point in points :
    print(point.text)

#driver.close()
