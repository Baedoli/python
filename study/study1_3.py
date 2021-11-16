
# 사전 실습

#cabinet = {3:"oh", 100:"bae"}
#print(cabinet[3])
#print(cabinet.get(100))
#print(cabinet.get(5,"Use Enable"))

#print(3 in cabinet)
#print(5 in cabinet)

cabinet = {"A-3":"oh", "A-100":"bae"}
print(cabinet["A-3"])
print(cabinet.get("A-100"))

#New
cabinet["A-3"] = "Mijin"
cabinet["B-100"] = "Yejin"

print(cabinet)
del cabinet["A-3"]
print(cabinet)

# Key 맟 출력
print(cabinet.keys())
print(cabinet.values())

# 쌍으로 출력
print(cabinet.items())

cabinet.clear()

