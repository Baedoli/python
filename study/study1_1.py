
jumin = "740102-1899118"

print("설 별 : "+jumin[7])
print("년 도 : "+jumin[0:2]) # 0 번째 값부터 2 이전 까지 정보를 가져옴. 0,1,2
print(" 울 : "+jumin[2:4])
print(" 일자 : "+jumin[4:6])
print("생년월일 : "+jumin[0:6])

print("a"+"b") # + 기호를 사용하여 문자열 결합 시 붙여서 결합된다.
print("a","b") # , 기호를 사용하여 문자열 결합 시 한칸 띄어서 결합된다.

print("나는 20살 입니다.")
print("나는 %d살 입니다" % 20)

print("나는 파아썬을 좋아 해요")
print("나는 %s를 좋아해요" % "Python")

# \n 줄바꿈을 의미 한다.
# \" 는 무자열 안에 " 표시 한다.
print("백문이 불여일견 \n백견이 불여일타")

# 저는 "나도 코딩" 입니다.
print("저는 \"나도 코등\" 입니다.")

# \ 표시 하고자 할때 \\  로 입니다.
print("ㅊ:\\users\\temp")

# \r 은 커서를 맨 압프오 이동하는 명령어
print("Red Apple\rPine")

# \b : 백스페이스 역할
print("Redd\bApple")

# \t tab 을 의미
print("Red\tApple")

# 퀴즈 : 사이트별로 비밀번호를 만들어 주슨 프로그램을 작성 하시요
# 예) http://naver.com
# 규칙 1 : http:// 부분은 제외 => naver.com
# 규칙 2 : 처믕 만나는 점). 이후 부분은 제외 naver
# 규칙 3 : 남은 글자 중 처음 세자리 + 글자개수 + 글자 내 'e' 개수 + "!" 로 신규 비밀번호 설정

# url = "http://naver.com"
url = "http://www.daum.net"
my_str = url.replace("http://","")
print(my_str)
my_str = my_str[:my_str.index(".")]
print(my_str)
passwd = my_str[:3]+str(len(my_str))+str(my_str.count("e"))+"!"
print(passwd)
print("{0} 의 비밀번호는 {1} 입니다.".format(url, passwd))
