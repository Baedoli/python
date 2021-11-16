
# 출석 번호가 1,2,3,4,5 => 101, 102 ... 100을 붙이기로 함 ..
students = [1,2,3,4,5]
print(students)
students = [i+100 for i in students]
print(students)

# 학생 이름의 길이를 변환
students = ["Bae","oh","han"]
print(students)
students = [len(i) for i in students]
print(students)

# 학생이름의 대문자로 변환
students = ["Bae","oh","han"]
print(students)
students = [i.upper() for i in students ]
print(students)

 

