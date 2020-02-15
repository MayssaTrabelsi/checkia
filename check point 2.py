#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


df = pd.read_csv('train.csv', encoding= "ISO-8859-1")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.isnull().sum().sum()


# In[7]:


df.dropna(axis=0, how='any', thresh=None, inplace=False)


# In[8]:


df.tail()


# In[9]:


df.describe()


# In[10]:


import matplotlib.pyplot as plt 


# In[11]:


plt.plot(df['Age'],df['Pclass'], 'ro')


# In[12]:


import seaborn as sns 


# In[13]:


df.hist()
plt.show()


# In[14]:


sns.boxplot(data=df)


# In[15]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df["Sex"]=encoder.fit_transform(df[["Sex"]])
df


# In[16]:


df['Age'].fillna(df['Age'].mean(), inplace=True)
df


# In[17]:


plt.title(" diff ages")
plt.xlabel("Age")
df['Age'].plot.hist()


# In[18]:


sns.distplot(df['Age'],bins=10,hist=True,kde=True,color="black")


# In[19]:


sns.lmplot(x="Age",y="Sex",data=df)


# In[20]:


correlations = df.corr()
correlations


# In[21]:


g = sns.FacetGrid(df, col="Sex")
g.map(plt.hist, 'Age' , bins=20)
g.add_legend();


# In[22]:


def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

        )


# In[23]:


plot_correlation_map( df )


# In[24]:


df[['Pclass','Survived']].groupby(['Survived'], as_index=True).mean()


# In[33]:


df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df


# In[35]:


Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                      "Dr":         "Officer",

                    "Rev":        "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                   "Lady" :      "Royalty" , 

                  "the Countess": "Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Miss",

                    "Mlle":       "Miss",

                    "Miss" :      "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mrs",

                    "Mrs" :       "Mrs",

                    "Master" :    "Master"

                    }


# In[1]:


Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                      "Dr":         "Officer",

                    "Rev":        "Officer”,

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                   "Lady" :      "Royalty"

                  "the Countess": "Royalty",

                    "Dona":       "Royalty”,

                    "Mme":        "Miss",

                    "Mlle":       "Miss",

                    "Miss" :      "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mrs",

                    "Mrs" :       "Mrs

                    "Master" :    "Master"

                    }

