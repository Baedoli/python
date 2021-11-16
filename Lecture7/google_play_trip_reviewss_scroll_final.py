# 팩키지 임포트
from selenium import webdriver
from datetime import datetime

import time
import pandas as pd
import os
# Excel 저장 할 경우 필요한 Package ...
import openpyxl

def extract_content(sections,dataset):
    counter = 1
    for section in sections:
        member = section.find_element_by_class_name("X43Kjb").text
        # 별점 Div 항목 안에 있는 값을 가져옴 ...
        rating = section.find_element_by_class_name("pf5lIe").find_element_by_tag_name("div").get_attribute(
            "aria-label")
        created = section.find_element_by_class_name("p2TkOb").text
        likes = section.find_element_by_class_name("jUL89d.y92BAb").text
        body_ = section.find_element_by_class_name("UD7Dzf").text
        counter = counter + 1

        example = [member, rating, created, likes, body_]
        example_series = pd.Series(example, index=dataset.columns)
        dataset = dataset.append(example_series, ignore_index=True)

    return dataset

driver_path = '/Users/baeseongho/webdriver/chromedriver'
driver = webdriver.Chrome(driver_path)
url = "https://play.google.com/store/apps/details?id=com.titicacacorp.triple&hl=ko&gl=US&showAllReviews=true"
driver.get(url)

column_names = ["member", "rating", "created", "likes", "comment"]
dataset = pd.DataFrame(columns = column_names)
dataset.head()

# Initialize counter
counter = 1
# driver.find_element_by_class_name('RveJvd.snByac').click()
SCROLL_PAUSE_TIME = 2
# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        sections = driver.find_elements_by_class_name("d15Mdf.bAhLNe")
        dataset = extract_content(sections, dataset)
        break
    last_height = new_height
    counter = counter + 1
    print("Counter: ", counter)
    # 3번 까지만 reload 하고 그 이상은 skip 한다 ...
    if counter > 10:
        # 교수님 코드
        # sections = driver.find_elements_by_class_name("d15Mdf.bAhLNe")
        # dataset = extract_content(sections, dataset)
        break

base_dir = "/Users/baeseongho/webdriver"
file_name = "google_play_trip_comment.xlsx"
xlxs_dir = os.path.join(base_dir, file_name)
dataset.to_excel(xlxs_dir, sheet_name = 'Sheet1', na_rep = 'NaN',\
                 float_format = "%.2f", header = True, startrow = 0, startcol = 0,\
                 engine = 'openpyxl')

