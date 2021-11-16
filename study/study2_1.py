
# 분기 실습

# weather = input("오늘 날시는 어때요?")
#
# if weather == "비" or weather == "눈":
#     print("우산을 챙기세요")
# elif weather == "바람":
#     print("미세먼지를 조심하세요")
# else:
#     print("준비물이 필요 없어요")
#

temp = int(input("기온은 몇도 에요?"))

if 30 <= temp:
    print("너무 더워요 나가지 마세요")
elif 10 <= temp and temp < 30:
    print("괜찮은 날씨에요")
elif 0 <= temp and temp < 10:
    print("날씨가 추워요")
else:
    print("나가지 마세요")




