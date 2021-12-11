import pandas as pd

# dictionary with list object in values
details = {
    'Name': ['Ankit', 'Aishwarya', 'Shaurya',
             'Shivangi', 'Priya', 'Swapnil'],
    'Age': [23, 21, 22, 21, 24, 25],
    'University': ['BHU', 'JNU', 'DU', 'BHU',
                   'Geu', 'Geu'],
}

# creating a Dataframe object
df = pd.DataFrame(details, columns=['Name', 'Age',
                                    'University'],
                  index=['a', 'b', 'c', 'd', 'e', 'f'])

print(df)
# get names of indexes for which column Age has value >= 21
# and <= 23
index_names = df[(df['Age'] >= 21) & (df['Age'] <= 23)].index

# drop these given row
# indexes from dataFrame
df.drop(index_names, inplace=True)

df.loc[df.Age>=25, 'result'] = 'old'
df.loc[df.Age<25, 'result'] = 'Young'

#df.loc[df.grades>50,'result']='success'

#df.head()
print(df)
