# 팩키지 임포트
from selenium import webdriver
from datetime import datetime

import time
import pandas as pd
import os
# Excel 저장 할 경우 필요한 Package ...
import openpyxl

def extract_content(sections,dataset,counter):
    for section in sections:
        member = section.find_element_by_class_name("X43Kjb").text
        # 별점 Div 항목 안에 있는 값을 가져옴 ...
        rating = section.find_element_by_class_name("pf5lIe").find_element_by_tag_name("div").get_attribute("aria-label")
        created = section.find_element_by_class_name("p2TkOb").text
        likes = section.find_element_by_class_name("jUL89d.y92BAb").text
        body_ = section.find_element_by_class_name("UD7Dzf").text

        # 등록자가 있는 정보만 가져 옴 ...
        if member != "":
            counter = counter + 1
            example = [member, rating, created, likes, body_]
            example_series = pd.Series(example, index=dataset.columns)
            dataset = dataset.append(example_series, ignore_index=True)

        # 최종 등록 건수가 5000개 이상이면 빠져 나옴 ..
        if counter > 5000:
            break

    return dataset, counter

driver_path = '/Users/baeseongho/webdriver/chromedriver'
driver = webdriver.Chrome(driver_path)

column_names = ["member", "rating", "created", "likes", "comment"]
dataset = pd.DataFrame(columns = column_names)
dataset.head()

# 알바천국
url = "https://play.google.com/store/apps/details?id=com.albamon.app&hl=ko&gl=US"

driver.get(url)

# 리뷰 모두 보기를 클릭 한다..
driver.find_element_by_xpath('//*[@id="fcxH9b"]/div[4]/c-wiz/div/div[2]/div/div/main/div/div[1]/div[6]/div/span/span').click()
SCROLL_PAUSE_TIME = 1.5

counter = 1
while True:

    # (1) 5번의 스크롤링
    for i in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        sections = driver.find_elements_by_class_name("d15Mdf.bAhLNe")
        dataset,counter = extract_content(sections, dataset,counter)
        print("현재 {0} 건이 담겼습니다.".format(counter))
        # 첫번째 Div 에 답긴 누적 건수가 5000 개 이상이면 종료 조건 ...
        if counter > 5000:
            break

    # 두번째 종료 조건 .. 최종 건수가 5000 개 이상이면 빠져 나간다..
    if counter > 5000:
        break

    # (2) 더보기 클릭
    try:
        driver.find_element_by_xpath("//*[@id='fcxH9b']/div[4]/c-wiz[2]/div/div[2]/div/div/main/div/div[1]/div[2]/div[2]/div/span/span").click()
    # 혹시나 네트워크 속도에 의거하여 느리게 될 경우 더보기 페이지를 찾지 못할경우 다시 시작 ...
    except:
        print("계속 진행 합니다.")
        continue

print("Excel 추출을 시작 합니다....")
base_dir = "/Users/baeseongho/webdriver"
# file_name = "mid_exam_2.xlsx"
file_name = "알바몬_리뷰_리스트.xlsx"
xlxs_dir = os.path.join(base_dir, file_name)
dataset.to_excel(xlxs_dir, sheet_name = 'Sheet1', na_rep = 'NaN',\
                 float_format = "%.2f", header = True, startrow = 0, startcol = 0,\
                 engine = 'openpyxl')
print("Excel 추출을 종료 합니다.")
