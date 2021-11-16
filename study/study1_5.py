
# Set : 집합

my_set = {1,2,3,3,3}
print(my_set)

java = {"mijin","yejin"}
python = set(["mijin", "bae","oh"])

print(java)
print(python)

# 교집합 java & python
print(java & python)
print(java.intersection(python))

# Java or python
print(java | python)
print(java.union(python))

# 차 집합 자바는 할 수 있으나 파이썬 은 핧수 없는 사람
print(java - python)
print(java.difference(python))

python.add("han")
print(python)