
# Import packages for analysis
import pandas as pd

# ata Preparation
# Load Data
data_path = '/Users/baeseongho/webdriver/corpus.txt'
data = open(data_path, encoding='utf8').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    result = line.strip()
    content = result.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

# Create data frame for text and label
dataset_df = pd.DataFrame()
dataset_df['text'] = texts
dataset_df['label'] = labels
dataset_df.head()

# creating instance of labelencoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
dataset_df['label'] = labelencoder.fit_transform(dataset_df['label'])
print(dataset_df['label'])

# Map original label to number
integer_mapping = {l: i for i, l in enumerate(labelencoder.classes_)}
print(integer_mapping)

from sklearn.model_selection import train_test_split
sentences_train, sentences_test, y_train, y_test \
    = train_test_split(dataset_df['text'],
                       dataset_df['label'] ,
                       test_size=0.25, random_state=1000)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
print(tokenizer.word_index)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
# index 0 번지는 다른 용도로 사용 하므로 +1 로 해준다 ..
vocab_size = len(tokenizer.word_index) + 1

# Adding 1 because of reserved 0 index
print(sentences_train[8])
print(X_train[8])

from tensorflow.keras.preprocessing.sequence import pad_sequences

# 임의의 maxlength 를 지정 ..
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(sentences_train[8])
print(X_train[8])

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
embedding_dim = 50
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plot_history(history)
