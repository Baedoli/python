
# 팩키지 임포트
from selenium import webdriver
from datetime import datetime

import time
import pandas as pd
import os
# Excel 저장 할 경우 필요한 Package ...
import openpyxl

driver_path = '/Users/baeseongho/webdriver/chromedriver'
driver = webdriver.Chrome(driver_path)
url = "https://play.google.com/store/apps/details?id=com.titicacacorp.triple&hl=ko&gl=US&showAllReviews=true"
driver.get(url)

column_names = ["member", "rating", "created", "likes", "comment"]
dataset = pd.DataFrame(columns = column_names)
dataset.head()

# def extract_content(sections, dataset):
counter = 1
sections = driver.find_elements_by_class_name("d15Mdf.bAhLNe")
for section in sections:
    if counter == 1 :
        print("***********************************************************")
    # print(section)
    member = section.find_element_by_class_name("X43Kjb").text
    print("작성자 :",member)
    # 별점 Div 항목 안에 있는 값을 가져옴 ...
    rating = section.find_element_by_class_name("pf5lIe").find_element_by_tag_name("div").get_attribute("aria-label")
    print("별점 :",rating)
    created = section.find_element_by_class_name("p2TkOb").text
    print("작성일자 :",created)
    likes = section.find_element_by_class_name("jUL89d.y92BAb").text
    print("좋아요 :",likes)
    body_ = section.find_element_by_class_name("UD7Dzf").text
    print("본문 :",body_)

    print("comment count :",counter)
    print("***********************************************************")
    counter = counter + 1

    example = [member, rating, created, likes, body_]
    example_series = pd.Series(example, index=dataset.columns)
    dataset = dataset.append(example_series, ignore_index=True)

dataset.head()

base_dir = "/Users/baeseongho/webdriver"
file_name = "google_play_trip_comment.xlsx"
xlxs_dir = os.path.join(base_dir, file_name)
dataset.to_excel(xlxs_dir, sheet_name = 'Sheet1', na_rep = 'NaN',\
                 float_format = "%.2f", header = True, startrow = 0, startcol = 0,\
                 engine = 'openpyxl')





