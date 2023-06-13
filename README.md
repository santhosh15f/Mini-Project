# DS_MINI_PROJECT
## Aim 
To Perform Data Visualization on IMDB dataset and save the data to a file.

## Explanation 
Data science is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.
## Algorithm 
### STEP 1:
Read the given Data
### STEP 2:
Clean the Data Set using Data Cleaning Process
### STEP 3:
Apply Feature generation and selection techniques to all the features of the data set
### STEP 4:
Apply data visualization techniques to identify the patterns of the data.

## Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
from scipy import stats

df=pd.read_csv("\imdb_top250_movies.csv")
df.head()
df['Year'].unique()
df = df.drop(['Unnamed: 0'],axis='columns')
df.info()
df.isnull().sum()
df.describe()
df.shape

df.groupby('Year')['Year'].agg('count').sort_values(ascending = False)
df1 = df.copy()
df1['imdbVotes'] = df1['imdbVotes'].apply(lambda x: x.replace(',',''))
df1['imdbVotes'] = df1['imdbVotes'].astype('int32')
df1['imdbVotes']

df1['Runtime'] = df1['Runtime'].apply(lambda x: x.replace(' min',''))
df1['Runtime'] = df1['Runtime'].apply(lambda x: x.strip())
df1['Runtime'] = df1['Runtime'].astype('int32')

df1['Genre'] = [i.split(',') if ',' in i else [i] for i in df.Genre.values]
df1['Actors'] = [i.split(',') if ',' in i else [i] for i in df.Actors.values]

df1["Released"]=df1["Released"].fillna(df1["Released"].mode()[0])
df1["Writer"]=df1["Writer"].fillna(df1["Writer"].mode()[0])
df1["Awards"]=df1["Awards"].fillna(df1["Awards"].mode()[0])
df1["DVD"]=df1["DVD"].fillna(df1["DVD"].mode()[0])
df1["Metascore"]=df1["Metascore"].fillna(df1["Metascore"].mode()[0])
df1["BoxOffice"]=df1["BoxOffice"].fillna(df1["BoxOffice"].mode()[0])

df1.isnull().sum()

df1.describe()

plt.figure(figsize=(16,10))
sns.boxplot(x="imdbRating",y="imdbVotes",data=df1)

plt.figure(figsize=(16,10))
sns.boxplot(x="imdbRating",y="Metascore",data=df1)

cols = ['imdbVotes','imdbRating','Metascore','Genre']
Q1 = df1[cols].quantile(0.25)
Q3 = df1[cols].quantile(0.75)
IQR = Q3 - Q1
df2 = df1[~((df1[cols] < (Q1 - 1.5 * IQR)) |(df1[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

df2.head()

plt.title("Dataset after removing outliers")
df2.boxplot()
plt.show()

plt.figure(figsize=(16,10))
sns.boxplot(x="imdbRating",y="imdbVotes",data=df2)

plt.figure(figsize=(16,10))
sns.boxplot(x="imdbRating",y="imdbVotes",data=df2)

plt.figure(figsize=(16,10))
sns.boxplot(x="imdbRating",y="Metascore",data=df2)

df2["imdbRating"].value_counts()

plt.figure(figsize=(16,10))
sns.countplot(x="imdbRating",data=df2)

plt.figure(figsize=(16,10))
sns.distplot(df2["imdbRating"],color="#6ac8e1")

plt.figure(figsize=(16,10))
sns.histplot(x="imdbRating",data=df2,color="#6ac8e1")

plt.figure(figsize=(64,40))
sns.countplot(x ='imdbVotes', data = df2)
plt.xticks(rotation = 50)
plt.show()

plt.figure(figsize=(16,10))
sns.distplot(df2["imdbVotes"],color="#6ac8e1")

plt.figure(figsize=(16,10))
sns.histplot(x="imdbVotes",data=df2,color="#6ac8e1")

plt.figure(figsize=(16,10))
sns.countplot(x ='Metascore', data = df2)

plt.figure(figsize=(16,10))
sns.distplot(df2["Metascore"],color="#6ac8e1")

plt.figure(figsize=(16,10))
sns.histplot(x="Metascore",data=df2,color="#6ac8e1")

plt.figure(figsize=(32,20))
sns.countplot(x ='Year', data = df2)
plt.xticks(rotation = 50)
plt.show()

plt.figure(figsize=(16,10))
sns.distplot(df2["Year"],color="#6ac8e1")

plt.figure(figsize=(16,10))
sns.histplot(x="Year",data=df2,color="#6ac8e1")

df2.corr()

ohe=OneHotEncoder(sparse=False)
df2["Country"] = ohe.fit_transform(df2[["Country"]])
df2["Language"] = ohe.fit_transform(df2[["Language"]])

df2.info()
df4=df2[["Num","Year","Language","Country","Metascore","imdbRating","imdbVotes"]]
df4

from sklearn.preprocessing import RobustScaler
robust_scaler=RobustScaler()
df5=pd.DataFrame(robust_scaler.fit_transform(df4))
df5.head()

film_releases = df4.groupby('Year')['Year'].agg('count')
year = df4['Year'].unique()
plt.figure(figsize=(16,10))
plt.bar(year,film_releases)
plt.xlabel("Year")
plt.ylabel("No. of film released")
plt.show()

null_percentage = df.isnull().sum() * 100 / len(df)
null_percentage

plt.figure(figsize=(16,10))
sns.distplot(null_percentage,color="#6ac8e1")

genre = {}
for i in df2['Genre'].values:
    for u in i:
        genre.setdefault(u, 0)
        genre[u]+=1

genre = dict(sorted(genre.items(), key=lambda item: item[1]))

plt.figure(figsize=(18, 10))
plt.plot([i for i in genre], [genre[i] for i in genre], linewidth=4.0)
 
plt.title("Most popular genre")
plt.ylabel('Count of films')
plt.xlabel("Genre")
plt.xticks(rotation=50, fontsize=12)

plt.show()

df2[df2['Runtime'] >= 180]['Title']

df4.groupby('Year')['imdbVotes'].mean().sort_values(ascending = False)

plt.figure(figsize=(16,10))
sns.barplot(x='Year', y='imdbVotes', data = df2)
plt.xticks(rotation = 90)
plt.title("Votes By Year")
plt.show()

df2.groupby('Director')['imdbRating'].mean().sort_values(ascending = False)

top10votes = df2.nlargest(10, 'imdbVotes')[['Title','imdbVotes']].set_index('Title')
top10votes

plt.figure(figsize=(10,8))
sns.barplot(x = 'imdbVotes', y = top10votes.index, data = top10votes)

plt.show()

genre = {}
for i in df2['Genre'].values:
    for u in i:
        genre.setdefault(u, 0)
        genre[u]+=1

genre = dict(sorted(genre.items(), key=lambda item: item[1]))

plt.figure(figsize=(18, 10))
plt.plot([i for i in genre], [genre[i] for i in genre], linewidth=4.0)
 
plt.title("Most popular genre")
plt.ylabel('Count of films')
plt.xlabel("Genre")
plt.xticks(rotation=50, fontsize=12)

plt.show()

df["Actors"].unique()

actors = {}
for i in df1['Actors'].values:
    for u in i:
        actors.setdefault(u, 0)
        actors[u]+=1

actors = dict(sorted(actors.items(), key=lambda item: item[1])[-30:])

plt.figure(figsize=(18, 10))
plt.bar([i for i in actors], [actors[i] for i in actors])
 
plt.title("Most popular actors")
 
plt.ylabel('Count of films')

plt.xticks(rotation=50, fontsize=12)

plt.show()

actor = {};
for i in df1[['Title','Actors', 'imdbRating']].values:
    if 'Hugh Jackman' in i[1]: 
        actor[i[0]] = i[2]

sns.set_theme(palette="pastel", font="arial", font_scale= 2.5)

plt.pie([actor[i] for i in actor], labels=[i for i in actor], autopct='%.2f',shadow=True)
plt.title("Popular Hugh Jackman films")

plt.rcParams["figure.figsize"] = (20,20)
plt.show()

actor = {};
for i in df1[['Title','Actors', 'imdbRating']].values:
    if 'Leonardo DiCaprio' in i[1]: 
        actor[i[0]] = i[2]

sns.set_theme(palette="pastel", font="arial", font_scale= 2.5)

plt.pie([actor[i] for i in actor], labels=[i for i in actor], autopct='%.2f',shadow=True)
plt.title("Popular Leonardo DiCaprio films")

plt.rcParams["figure.figsize"] = (20,20)
plt.show()

production = {}
for i in df1['Production'].values:
    production.setdefault(i, 0)
    production[i]+=1

production = dict(sorted(production.items(), key=lambda item: item[1])[-20:])

plt.figure(figsize=(20, 8))
plt.bar([i for i in production], [production[i] for i in production])
 
plt.title("Most popular productions")
 
plt.ylabel('Count of films')

plt.xticks(rotation=80, fontsize=12)

plt.show()

Top = df2[['Title', 'Metascore']].sort_values(by="Metascore",ascending=False)
Top = Top.head(20)
Top.head(10)

plt.figure(figsize=(18,10))
graph=sns.barplot(y='Metascore',x='Title',data=Top)
graph.set_title('Top 20 Movies By Metascore')
plt.xticks(rotation=85)
plt.show()

dfDirectors = df2[["Director"]].apply(pd.value_counts)
dfDirectors = dfDirectors.head(20)
dfDirectors.plot(kind="bar", figsize=(18,10))
plt.xticks(rotation = 85)
plt.title("Top 20 Popular Directors")
plt.show()

mRuntym=df2[['Runtime', 'imdbRating']].sort_values(by="Runtime",ascending=False)
mRuntym.head()

plt.figure(figsize=(16,10))
sns.set_theme(font_scale= 1)
graph=sns.lineplot(x='Runtime',y='imdbRating',data=mRuntym)
graph.set_title('Runtime Vs IMDB Rating')
plt.xticks() 
plt.show()

plt.figure(figsize=(80,50))
sns.set_theme(font_scale= 1)
sns.pointplot(y=mRuntym["imdbRating"],x=mRuntym["Runtime"])
plt.xticks(rotation = 85)
plt.show()

mbox=df2[['BoxOffice', 'imdbRating']].sort_values(by="BoxOffice",ascending=False)
mbox.head()

plt.figure(figsize=(16,10))
sns.set_theme(font_scale= 1)
graph=sns.lineplot(y='BoxOffice',x='imdbRating',data=mbox)
graph.set_title('Box Office Vs IMDB Rating')
plt.xticks() 
plt.show()

plt.figure(figsize=(40,25))
sns.set_theme(font_scale= 1)
sns.pointplot(y=mbox["imdbRating"],x=mbox["BoxOffice"])
plt.xticks(rotation = 85)
plt.show()

plt.figure(figsize=(16,10))
sns.heatmap(df2.corr(),cmap="YlGnBu",annot=True)
```
## Output
![](img/1.JPG)
![](img/2.JPG)
![](img/3.JPG)
![](img/4.JPG)
![](img/5.JPG)
![](img/6.JPG)
![](img/7.JPG)
![](img/8.JPG)
![](img/9.JPG)
![](img/10.JPG)

![](img/11.JPG)
![](img/12.JPG)
![](img/13.JPG)
![](img/14.JPG)
![](img/15.JPG)
![](img/16.JPG)
![](img/17.JPG)
![](img/18.JPG)
![](img/19.JPG)
![](img/20.JPG)

![](img/21.JPG)
![](img/22.JPG)
![](img/23.JPG)
![](img/24.JPG)
![](img/25.JPG)
![](img/26.JPG)
![](img/27.JPG)
![](img/28.JPG)
![](img/29.JPG)
![](img/30.JPG)

![](img/31.JPG)
![](img/32.JPG)
![](img/33.JPG)
![](img/34.JPG)
![](img/35.JPG)
![](img/36.JPG)
![](img/37.JPG)
![](img/38.JPG)
![](img/39.JPG)
![](img/40.JPG)

![](img/41.JPG)
![](img/42.JPG)
![](img/43.JPG)
![](img/44.JPG)
![](img/45.JPG)
![](img/46.JPG)
![](img/47.JPG)
![](img/48.JPG)
![](img/49.JPG)
![](img/50.JPG)

![](img/51.JPG)
## Result
Thus we have successfully visualized the dataset using libraries like pandas, seaborn and so on. 
