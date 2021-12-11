
cities = ['London', 'Berlin', 'Berlin', 'New York', 'London']

# LabelEncoder : 문자형을 정수형으로 변환 하는 함수 ..
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
city_labels = encoder.fit_transform(cities)
print(city_labels)

encoder_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print(encoder_name_mapping)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
city_labels = city_labels.reshape((5, 1))
city_labels = encoder.fit_transform(city_labels)
print(city_labels)

city_labels = encoder.fit_transform(city_labels)
print(city_labels)

