
from random import *
cnt = 0  # 총 탑승자 수
for i in range(1, 51):   # 1 ~ 50명 승객 탑승 ..
    time = randrange(5,51)   # 5 ~ 50 random 으로 가져옴 ..

    if time >= 5 and time <= 15: # 탑승 조건 소요 시간 5분 ~ 15분 사이일 경우 ...
        print("[0] {0} 번째 손님 ( 소요시간 : {1}분 )".format(i, time))
        cnt += 1
    else:
        print("[ ] {0} 번째 손님 ( 소요시간 : {1}분 )".format(i, time))

print("총 탓승 손님 수 : {0}".format(cnt))


 