
movies = ["The1", "The2", "The3"]

movies1 = ["The Holy Grail",1975, "Terry Jones & Terry Gilliam", 91,
           ["Graham Chapma", ["Michael Palin", "John Cleese", "Terryu Gilliam", "Eric Idle", "Terry Jones"]]]

print(len(movies))
print(movies[1])

for list_movie in movies:
    print(list_movie)

count_ = 0
while count_ < len(movies):
    print(movies[count_])
    count_ = count_ + 1

print(isinstance(movies,list))

leng_movies = len(movies)

print(isinstance(leng_movies,list))

for each_item in movies1:
    if isinstance(each_item,list):
        for each_item1 in each_item:
            if isinstance(each_item1,list):
                for each_item2 in each_item1:
                    print(each_item2)
            else:
                print(each_item1)
    else:
        print(each_item)

def print_arry (list_arry):

    for each_item in list_arry:
        if isinstance(each_item,list):
            print_arry(each_item)
        else:
            print(each_item)

print_arry(movies1)



