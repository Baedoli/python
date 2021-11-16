
# List Test

subway = ["oh","bae"]
print(subway.index("oh"))
# 추가
subway.append("kim")
print(subway)
# 특정 위치 삽입
subway.insert(1,"park")
print(subway)

# 내림 ...
print(subway.pop())
print(subway)

#같은 사람이 몇명 ...
subway.append("oh")
print(subway)

# 리스트 내 항목 카운트
print(subway.count("oh"))

# 리스트 내 정렬 하기
num_list = [5,3,4,2,1]
num_list.sort()
print(num_list)

#순서 뒤집기 reverse
num_list.reverse()
print(num_list)

#리스트 지우기
# num_list.clear()
print(num_list)

#다양한 자료형 가능
mix_list = ["oh", 2, True]
print(mix_list)

# 두개 list 결합 하기
num_list.extend(mix_list)
print(num_list)