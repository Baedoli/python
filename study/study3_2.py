
# 함수 호출 시 기본값 셋팅 방법 ....
def profile(name, age=17, main_lang="python"):
    print("이름 : {0}\t나이 : {1}\t주 사용 언어: {2}".format(name, age, main_lang))

def profile_new(name, age, lang1, lang2, lang3, lang4, lang5):
    print("이름 : {0}\t나이 : {1}\t".format(name, age), end="  ")
    print(lang1, lang2, lang3, lang4, lang5)

def profile_var(name, age, *language):
    print("이름 : {0}\t나이 : {1}\t".format(name, age),end="   ")
    for lang in language:
        print(lang,end="  ")
    print()

profile("배성호")
profile("오정임", 30, "Java")

profile(name="배미진")
profile(age=16,name="배예진",main_lang="PL-SQL")

profile_new("배성호", 20, "Python","java","C","C++","C#")
profile_new("오정임",45,"PL-SQL","","","","")

profile_var("배성호", 20, "Python","java","C","C++","C#","JavaScript")
profile_var("오정임",45,"PL-SQL")