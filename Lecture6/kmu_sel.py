
# 팩키지 임포트
from selenium import webdriver
from datetime import datetime

#  driver라는 이름의 webdriver 객체를 만들어 주자.
#  Chrome의 경우 | 아까 받은 chromedriver의 위치를 지정해준다.

driver_path = '/Users/baeseongho/webdriver/chromedriver'
driver = webdriver.Chrome(driver_path)
driver2 = webdriver.Chrome(driver_path)

# kmu notification url
url = 'https://www.kmu.ac.kr/uni/main/page.jsp?mnu_uid=143&';
driver.get(url)

posts = driver.find_elements_by_tag_name("tr")

line_count = 1
for post in posts :
    # 일련번호 찾기 ...
    # class name tag name 에 공백이 있을 경우 . 으로 표시해야만 오류가 나지 않는다....

    if line_count > 1:
        num_ = post.find_element_by_class_name("num.first").text
        subject_ = post.find_element_by_class_name("subject").text
        # 해당 제목안에 있는 anchor tag 안에 있는 href 속성값을 가져 오면 해당 url 정보를 가져온다....
        subject_str = post.find_element_by_class_name("subject").find_element_by_tag_name("a").get_attribute("href")
        driver2.get(subject_str)

        body_ =  driver2.find_element_by_class_name("bbs_con").text
        print(body_)

        # writer_ = post.find_element_by_class_name("writer").text
        # date_ = post.find_element_by_class_name("date").text
        # hit_last_ = post.find_element_by_class_name("hit.last").text
        # print(num_)
        # print(subject_)
        # print(writer_)
        # print(date_)
        # print(hit_last_)

    else:
        line_count = line_count + 1

