#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 
import nltk
from nltk.tokenize import TweetTokenizer
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from wordcloud import WordCloud
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.metrics import log_loss 
from sklearn.feature_extraction.text import TfidfVectorizer
#import re
import datetime
#import numpy as np
#import pandas as pd
#from random import shuffle
#import matplotlib.pyplot as plt
#from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
#from numpy import loadtxt
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#import xgboost as xgb
#from sklearn.model_selection import KFold, train_test_split, GridSearchCV
#from sklearn.metrics import confusion_matrix, mean_squared_error
#from sklearn.datasets import load_boston
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#import time
#from sklearn.metrics import accuracy_score
#from keras.models import load_model
#import nltk
#import seaborn as sns
#from nltk.tokenize import TweetTokenizer
#import string
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LassoCV, ElasticNetCV
#from sklearn import tree
#import itertools
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LogisticRegressionCV
#from scipy import stats
#from sklearn import svm
#from sklearn.decomposition import PCA
#from sklearn.svm import SVR
#import xgboost as xgb
#from keras.layers.core import Dense
#from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
#from keras.layers.normalization import BatchNormalization
#from keras.layers import Dense, Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras import optimizers
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
#from sklearn.linear_model import ElasticNet
#from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


# In[38]:


train_csv=pd.read_csv('train.csv')
test_csv=pd.read_csv('test.csv')


# In[39]:


train_csv.head()


# ## Data Processing and EDA 

# In[40]:


train_csv.shape


# In[41]:


train_csv.isnull().sum()


# In[42]:



missing_rows= (train_csv.isnull().sum().count()*100) /len(train_csv)
missing_rows


# In[43]:


train_csv.info()


# In[44]:


train_csv['price'] = train_csv['price'].replace('[\$\,\.]', '', regex=True).astype(int)
train_csv['price']=train_csv['price']/100


# In[45]:


describe_train_csv = train_csv.describe()
describe_train_csv.loc['skewness']= train_csv.skew()
describe_train_csv.loc['kurtosis']= train_csv.kurt()
describe_train_csv.round(3)


# In[46]:


response = train_csv['price'].describe()
response.loc['skewness']= train_csv['price'].skew()
response.loc['kurtosis']= train_csv['price'].kurt()
response.round(3)


# In[47]:


from statlearning import plot_dist
plot_dist(train_csv['price'])
plt.title('Distribution of Price')
plt.show()


# In[48]:


plot_dist(train_csv["latitude"])
plt.title('Distribution of latitude')
plt.show()


# In[49]:


plot_dist(train_csv["longitude"])
plt.title('Distribution of longitude')
plt.show()


# In[50]:


# These next two lines shouldn't be necessary, but they fix an error that I'm getting at the moment.
# This will be different for your computer, if you need it
import os
# os.environ['PROJ_LIB'] = r'C:\Users\xps\Anaconda3\Library\share'
os.environ["PROJ_LIB"] = '/Users/zhuohanzhu/anaconda3/pkgs/proj4-5.2.0-h0a44026_1/share/proj'
from mpl_toolkits.basemap import Basemap

def california_map(ax=None, lllat=-34.088, urlat=-33.390,
                   lllon=150.601,urlon=151.340):
# This function is based on "train_csv Analytics Using Open-Source Tools" by Jeffrey Strickland
    
    m = Basemap(ax=ax, projection='stere',
                lon_0=(urlon + lllon) / 2,
                lat_0=(urlat + lllat) / 2,
                llcrnrlat=lllat, urcrnrlat=urlat,
                llcrnrlon=lllon, urcrnrlon=urlon, resolution='f')
    m.drawstates()
    m.drawcountries()
    m.drawcoastlines(color='lightblue')
    return m


# Plot Figure
fig, ax = plt.subplots(figsize=(9,9))
m = california_map()
x, y = m(train_csv['longitude'].as_matrix(), train_csv['latitude'].as_matrix())

cmap = sns.diverging_palette(220, 10, as_cmap=True)
# m.scatter(x,y,s=train_csv['accommodates']), c=train_csv['price'], edgecolors='none', cmap=plt.get_cmap('rainbow'),
#          alpha=0.5)
m.scatter(x,y,s=train_csv['number_of_reviews']/5, c=train_csv['price'], edgecolors='none', cmap=plt.get_cmap('rainbow'), alpha=0.5)

ax.set_title('Airbnb house number_of_reviews', fontsize=17, y=1.01, fontweight='bold')
ax.spines['bottom'].set_color('#DDDDDD')
ax.spines['top'].set_color('#DDDDDD')
ax.spines['right'].set_color('#DDDDDD')
ax.spines['left'].set_color('#DDDDDD')

plt.tight_layout()
plt.show()


# In[51]:


train_csv.corr()


# In[52]:


train_csv.corr()["price"].abs().sort_values()


# In[53]:


locations = ['latitude', 'longitude']
from statlearning import plot_regressions
plot_regressions(train_csv[locations], train_csv['price'], lowess=True)
plt.show()


# ## Feature Engineering

# In[54]:


train=train_csv.loc[:, ['name','summary','space'
                                 , 'neighborhood_overview', 'notes',
                                 'transit', 'access', 'interaction', 'house_rules', 'host_since','host_about',
                                 'host_response_time', 'host_response_rate', 'host_is_superhost','host_listings_count',
                                 'host_verifications', 'host_identity_verified',
                                 'neighbourhood_cleansed', 'latitude', 'longitude', 'is_location_exact','property_type',
                                 'room_type', 'accommodates', 'bathrooms', 'beds','bedrooms','bed_type', 'amenities',
                                 'weekly_discount', 'monthly_discount', 'security_deposit_perc',
                                 'cleaning_fee_perc', 'guests_included', 'extra_people_perc',
                                 'minimum_minimum_nights', 'maximum_maximum_nights',
                                 'availability_365','number_of_reviews', 'review_scores_rating',
                                  'review_scores_accuracy', 'first_review','last_review','review_scores_checkin',
                                 'review_scores_communication', 'review_scores_location',
                                 'review_scores_value', 'instant_bookable','require_guest_profile_picture','require_guest_phone_verification',
                                 'cancellation_policy', 'reviews_per_month','review_scores_cleanliness',
                                 'calculated_host_listings_count','calculated_host_listings_count_entire_homes',
                                 'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms','price']]


# In[55]:


test=test_csv.loc[:, ['name','summary','space'
                                 , 'neighborhood_overview', 'notes',
                                 'transit', 'access', 'interaction', 'house_rules', 'host_since','host_about',
                                 'host_response_time', 'host_response_rate', 'host_is_superhost','host_listings_count',
                                 'host_verifications', 'host_identity_verified',
                                 'neighbourhood_cleansed', 'latitude', 'longitude', 'is_location_exact','property_type',
                                 'room_type', 'accommodates', 'bathrooms', 'beds','bedrooms','bed_type', 'amenities',
                                 'weekly_discount', 'monthly_discount', 'security_deposit_perc',
                                 'cleaning_fee_perc', 'guests_included', 'extra_people_perc',
                                 'minimum_minimum_nights', 'maximum_maximum_nights',
                                 'availability_365','number_of_reviews', 'review_scores_rating',
                                  'review_scores_accuracy', 'first_review','last_review','review_scores_checkin',
                                 'review_scores_communication', 'review_scores_location',
                                 'review_scores_value', 'instant_bookable','require_guest_profile_picture','require_guest_phone_verification',
                                 'cancellation_policy', 'reviews_per_month','review_scores_cleanliness',
                                 'calculated_host_listings_count','calculated_host_listings_count_entire_homes',
                                 'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms']]


# In[56]:


data=pd.concat([train.iloc[:,:-1],test])
data=data.reset_index(drop=True)


# In[57]:


lastdate = datetime.datetime(2019, 7, 31,00,00)         # last date set 
host_since_past_days=[]
for i in data['host_since'][:9838]:
    duration = lastdate- datetime.datetime.strptime(i,'%d/%m/%y')
    duration=duration.days
    host_since_past_days.append(duration)

for i in data['host_since'][9838:]:
    duration = lastdate- datetime.datetime.strptime(i,'%Y-%m-%d')
    duration=duration.days
    host_since_past_days.append(duration)
    
data.drop('host_since',axis=1,inplace=True)
data = data.assign(host_since_days = host_since_past_days) 
data['host_since_days'].head()


# In[58]:


# we did feature engineering -- review gap
data["last_review"][:9838]=pd.to_datetime(data["last_review"][:9838],format='%d/%m/%y')

data["last_review"][9838:]=pd.to_datetime(data["last_review"][9838:],format='%Y-%m-%d')

data["first_review"][:9838]=pd.to_datetime(data["first_review"][:9838],format='%d/%m/%y')
data["first_review"][9838:]=pd.to_datetime(data["first_review"][9838:],format='%Y-%m-%d')

data['review_gap']=data["last_review"]-data["first_review"]
#data['review_gap'] = np.abs(data['review_gap'])
data['review_gap']=data['review_gap']/np.timedelta64(1, 'D')
#review_gap1=np.abs(data['review_gap'])
data['review_gapNA'] = data['review_gap'].isnull().astype(int) 
data['review_gap'] = data['review_gap'].fillna(0)
# since no last and first review time indicates there is no review, we therefore use a dummy to indicate null values


# In[59]:


days_since_first_review=[]
for i in data['first_review']:
    if pd.isnull(i):  # simply put i is None will leave NaT data type to "else"
        days_since_first_review.append(i)
    else:
        duration = lastdate- pd.to_datetime(i,format='%d/%m/%y') # need to type in format or there will be negative values
        duration=duration.days
        days_since_first_review.append(duration)
        

data = data.assign(days_since_first_review = days_since_first_review) 
data['days_since_first_review'].head()
data['days_since_first_reviewNA'] = data['days_since_first_review'].isnull().astype(int) 
data.drop('first_review',axis=1,inplace=True)
data['days_since_first_review'] = data['days_since_first_review'].fillna(0) 


# In[60]:


days_since_last_review=[]
for i in data['last_review']:
    if pd.isnull(i):  # simply put i is None will leave NaT data type to "else"
        days_since_last_review.append(i)
    else:
        duration = lastdate- pd.to_datetime(i,format='%d/%m/%Y') # need to type in format or there will be negative values
        duration=duration.days
        days_since_last_review.append(duration)
        
data = data.assign(days_since_last_review = days_since_last_review) 
data['days_since_last_review'].head()
data['days_since_last_reviewNA'] = data['days_since_last_review'].isnull().astype(int) 
data.drop('last_review',axis=1,inplace=True)
data['days_since_last_review'] = data['days_since_last_review'].fillna(0)


# In[61]:


#data['weekly_discountNA'] = np.abs(data['weekly_discount'])
data['weekly_discountNA'] = data['weekly_discount'].isnull().astype(int) 
data['weekly_discount'] = data['weekly_discount'].fillna(0) 

#data['monthly_discountNA'] = np.abs(data['monthly_discount'])
data['monthly_discountNA'] = data['monthly_discount'].isnull().astype(int) 
data['monthly_discount'] = data['monthly_discount'].fillna(0)


# In[62]:


data['host_response_rate'] = data['host_response_rate'].str.rstrip('%').astype('float')

data['host_response_rateNA'] = data['host_response_rate'].isnull().astype(int)
data['host_response_rate'] = data['host_response_rate'].fillna(0) 

data['security_deposit_percNA'] = data['security_deposit_perc'].isnull().astype(int) 
data['security_deposit_perc'] = data['security_deposit_perc'].fillna(0) 

data['cleaning_fee_percNA'] = data['cleaning_fee_perc'].isnull().astype(int) 
data['cleaning_fee_perc'] = data['cleaning_fee_perc'].fillna(0) 

data['review_scores_ratingNA'] = data['review_scores_rating'].isnull().astype(int) 
data['review_scores_rating'] = data['review_scores_rating'].fillna(0) 

data['review_scores_accuracyNA'] = data['review_scores_accuracy'].isnull().astype(int) 
data['review_scores_accuracy'] = data['review_scores_accuracy'].fillna(0) 

data['review_scores_cleanlinessNA'] = data['review_scores_cleanliness'].isnull().astype(int) 
data['review_scores_cleanliness'] = data['review_scores_cleanliness'].fillna(0) 

data['review_scores_checkinNA'] = data['review_scores_checkin'].isnull().astype(int) 
data['review_scores_checkin'] = data['review_scores_checkin'].fillna(0) 

data['review_scores_communicationNA'] = data['review_scores_communication'].isnull().astype(int) 
data['review_scores_communication'] = data['review_scores_communication'].fillna(0) 

data['review_scores_locationNA'] = data['review_scores_location'].isnull().astype(int) 
data['review_scores_location'] = data['review_scores_location'].fillna(0) 

data['review_scores_valueNA'] = data['review_scores_value'].isnull().astype(int) 
data['review_scores_value'] = data['review_scores_value'].fillna(0) 

data['reviews_per_monthNA'] = data['reviews_per_month'].isnull().astype(int) 
data['reviews_per_month'] = data['reviews_per_month'].fillna(0) 


# In[63]:


#distance to Sydney Opera House
lat_op=-33.856159
lon_op=151.215256
##manly beach
lat_ml=-33.7928
lon_ml=151.2763
##bondi beach
lat_bd=-33.890842
lon_bd=151.274292

def distance(lat1,lon1,lat2,lon2):
    return np.sqrt(np.square(lat1-lat2)+np.square(lon1-lon2))

dist_op=[]
dist_ml=[]
dist_bd=[]
for i in range(len(data)):
    lat1=data['latitude'][i]
    lon1=data['longitude'][i]
    dist_op.append(distance(lat1,lon1,lat_op,lon_op))
    dist_ml.append(distance(lat1,lon1,lat_ml,lon_ml))
    dist_bd.append(distance(lat1,lon1,lat_bd,lon_bd))
    
data['dist_op']=pd.Series(dist_op)
data['dist_ml']=pd.Series(dist_ml)
data['dist_bd']=pd.Series(dist_bd)
data = data.drop(['latitude', 'longitude'], axis=1) 


# In[64]:


data['superhost'] = (data['host_is_superhost'] == 't').astype(int) 
data['host_id_verified'] = (data['host_identity_verified'] == 't').astype(int) 
data['inst_bkb'] = (data['instant_bookable'] == 't').astype(int) 
data.drop(['host_is_superhost', 'host_identity_verified', 'instant_bookable'], axis=1,inplace=True)


# In[65]:


data['host_verification_level'] = data["host_verifications"].apply(lambda s: len(s.split(', '))) 
data['host_verification_level'].describe() 
data.drop(['host_verifications'], axis=1, inplace=True) 


# In[66]:


#To count the unique number of amentity category
data['amenities'].nunique() #9637


# In[67]:


#To count the number of amentities for each listing 
data['amenities_count'] = data["amenities"].apply(lambda s: len(str(s)[1:].split(',')))
data['amenities_count']
#data.drop(["amenities"], axis=1, inplace=True)


# In[68]:


keys = {
        'a few days or more': 1, 
        'within a day': 2, 
        'within a few hours': 3, 
        'within an hour': 4, 
}

data['host_response_time'] = data['host_response_time'].replace(keys)
data['response_time_NA'] = data['host_response_time'].isnull().astype(int)
data['host_response_time'] = data['host_response_time'].fillna(0)


# In[69]:


counts_prop = data['property_type'].value_counts()/len(data)
counts_accu=0
threshold=0.99
for i in range(0,len(counts_prop)):
    counts_accu=counts_accu+counts_prop[i]
    if counts_accu > threshold:
        print(counts_prop.index[i])
        break


# In[70]:


for level in counts_prop.index[i:]:
    data['property_type'][data['property_type']==level] = 'Other'

variable = 'property_type' 
dummies = pd.get_dummies(data[variable],  prefix = variable, drop_first=True) 
data = data.join(dummies) 

dummies.head()


# In[71]:


data.drop("property_type", axis=1,inplace=True)


# In[72]:


data["room_type"].value_counts()


# In[73]:


variable = 'room_type' 
dummies = pd.get_dummies(data[variable],  prefix = variable, drop_first=True) 
data = data.join(dummies)
dummies.head() 


# In[74]:


data.drop("room_type", axis=1,inplace=True)


# In[75]:


data['cancellation_policy'].value_counts()


# In[76]:


data["cancellation_policy"].unique()


# In[77]:


data.loc[data["cancellation_policy"]=="super_strict_60", "cancellation_policy"] = 'strict'
data.loc[data["cancellation_policy"]=="super_strict_30", "cancellation_policy"] = 'strict'
data.loc[data["cancellation_policy"]=="strict_14_with_grace_period", "cancellation_policy"] = 'strict'
data.loc[data["cancellation_policy"]=="luxury_super_strict_125", "cancellation_policy"] = 'strict'
data.loc[data["cancellation_policy"]=="luxury_no_refund", "cancellation_policy"] = 'strict'
data.loc[data["cancellation_policy"]=="luxury_moderate", "cancellation_policy"] = 'moderate'

#keys = {'flexible': 1, 
#       'moderate': 2, 
#        'strict': 3, 
#}

#data['cancellation_policy'] = data['cancellation_policy']. replace(keys) 

## we ignore other in this case then
variable = 'cancellation_policy' 
dummies = pd.get_dummies(data[variable],  prefix = variable, drop_first=True) 
data = data.join(dummies) 

dummies.head()


# In[78]:


data.drop("cancellation_policy", axis=1,inplace=True)


# In[79]:


data['is_location_exact'].value_counts()


# In[80]:


variable = 'is_location_exact' 
dummies = pd.get_dummies(data[variable],  prefix = 'exact_location', drop_first=True) 
data = data.join(dummies) 
dummies.head()


# In[81]:


data.drop("is_location_exact", axis=1,inplace=True)


# In[82]:


data['require_guest_profile_picture'].value_counts()


# In[83]:


variable = 'require_guest_profile_picture' 
dummies = pd.get_dummies(data[variable],  prefix = 'guest_pic_veri', drop_first=True)
data = data.join(dummies) 
dummies.head()


# In[84]:


data.drop("require_guest_profile_picture", axis=1,inplace=True)


# In[85]:


data['require_guest_phone_verification'].value_counts()


# In[86]:


variable = 'require_guest_phone_verification' 
dummies = pd.get_dummies(data[variable],  prefix = 'guest_phone_veri', drop_first=True) 
data = data.join(dummies) 
dummies.head()


# In[87]:


data.drop("require_guest_phone_verification", axis=1,inplace=True)


# In[88]:


data['bathrooms'].value_counts()


# In[89]:


data['bathrooms'].fillna(1,inplace = True) 


# In[90]:


## 1 is the most common value and it makes sense by examining the original data
data['bathrooms'].isnull().sum()


# In[91]:


data['bedrooms'].value_counts()


# In[92]:


# 1 is the most common value and it makes sense by examining the original data
data['bedrooms'].fillna(1,inplace = True) 


# In[93]:


data['bedrooms'].isnull().sum()


# In[94]:


data['beds'].value_counts()


# In[95]:


data['beds'].isnull().sum()


# In[96]:


data['beds'].fillna(data['bedrooms'],inplace=True)


# In[97]:


data['beds'].isnull().sum()


# In[98]:


bedtype = data['bed_type'].value_counts()


# In[99]:


for level in bedtype.index:
    if bedtype[level] <10:
        data['bed_type'][data['bed_type']==level] = 'Other'


# In[100]:


data['bed_type'].value_counts()


# In[101]:


variable = 'bed_type' 
dummies = pd.get_dummies(data[variable],  prefix = variable, drop_first=True) 
data = data.join(dummies)   
dummies.head()


# In[102]:


data.drop("bed_type", axis=1,inplace=True)


# In[103]:


data['neighbourhood_cleansed'].value_counts()


# In[104]:


variable = 'neighbourhood_cleansed' 
dummies = pd.get_dummies(data[variable],  prefix = variable, drop_first=True) 
data = data.join(dummies) 
dummies.head()


# In[105]:


data.drop("neighbourhood_cleansed", axis=1,inplace=True)


# In[106]:


data.drop(['name', 'space', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'host_about'], axis=1, inplace=True)


# In[107]:


data2=data.copy()
data_x=data2.iloc[:9838,:].drop(['amenities','summary'],axis=1)
test=data2.iloc[9838:,:].drop(['amenities','summary'],axis=1)


# In[108]:


data_x['summary']=data['summary'][:9838]
test['summary']=data['summary'][9838:]
#test['summary']=data['summary'][9838:].reset_index(drop=True)


# In[109]:


def process_text(text):
    Tokenizer = TweetTokenizer()
    tokenized = Tokenizer.tokenize(text)
    punctuation = list(string.punctuation)
    tokenized_no_punctuation=[word.lower() for word in tokenized if word not in punctuation]
    tokenized_no_stopwords=[word for word in tokenized_no_punctuation if word not in stopwords.words('english')]
    tokens = [PorterStemmer().stem(word) for word in tokenized_no_stopwords if word != '️']
    return tokens


# In[110]:


# Applies the process_text function separately to each element of the column 'text' 
data_x['tokens']=data_x['summary'].apply(process_text)   #[:500]


# In[111]:


fdist_train = nltk.FreqDist()
for words in data_x['tokens']:
    for word in np.unique(words):
            fdist_train[word] += 1           
#data_x['tokens'].to_hdf('train_tokens.h5', 'train_tokens')


# In[112]:


test['tokens']=test['summary'].apply(process_text)   #[:500]

fdist_test = nltk.FreqDist()
for words in test['tokens']:
    for word in np.unique(words):
            fdist_test[word] += 1
               
#test['tokens'].to_hdf('test_tokens.h5', 'test_tokens')


# In[113]:


summary_fdist_train = fdist_train.most_common()[:40]
summary_fdist_test = fdist_test.most_common()[:40]

for i in summary_fdist_train:
    data_x['summary_'+str(i[0])]= data_x["summary"].apply(lambda s: 1 if (str(i[0]) in s) else 0) 
# no "" around str(i[0])

for i in summary_fdist_test:
    test['summary_'+str(i[0])]= test["summary"].apply(lambda s: 1 if (str(i[0]) in s) else 0) 
# no "" around str(i[0])


# In[114]:


summary_train_col = [col for col in data_x if col.startswith('summary_')]
summary_test_col = [col for col in test if col.startswith("summary_")]

summary_train=data_x[summary_train_col]
summary_test=test[summary_test_col]


# In[115]:


amenities_list_train2 = data['amenities'][:9838] 
amenities_list_train = [[word.strip('[" ]') for word in row[1:-1].split(',')] for row in amenities_list_train2]


# In[116]:


freq_dist_amenities_train = nltk.FreqDist()  
for words in amenities_list_train:  
    for word in words:  
            freq_dist_amenities_train[word] += 1  

amenities_train=freq_dist_amenities_train.most_common()[:40]

for i in amenities_train:
    data_x['amenities_'+str(i[0])]= amenities_list_train2.apply(lambda s: int('str(i[0])' in str(s)[1:].split(','))) 


# In[117]:


amenities_list_test2 = data['amenities'][9838:] 
amenities_list_test = [[word.strip('[" ]') for word in row[1:-1].split(',')] for row in amenities_list_test2]  

freq_dist_amenities_test = nltk.FreqDist()  
for words in amenities_list_test:  
    for word in words:  
            freq_dist_amenities_test[word] += 1  

amenities_test=freq_dist_amenities_test.most_common()[:40]

for i in amenities_test:
    test['amenities_'+str(i[0])]= amenities_list_test2.apply(lambda s: int('str(i[0])' in str(s)[1:].split(','))) 


# In[118]:


# To reset the text index from 0
test = test.reset_index(drop=True) 


# In[119]:


# Delete different columns in data_x and test (predictors)
test_columns_del = []
for i in test.columns:
    if i not in data_x.columns:
        test_columns_del.append(i)
test_columns_del

train_columns_del = []
for i in data_x.columns:
    if i not in test.columns:
        train_columns_del.append(i)
train_columns_del

test = test.drop(test_columns_del,axis=1)
data_x = data_x.drop(train_columns_del,axis=1)


# In[120]:


print(data_x.shape) 
print(test.shape)   


# In[121]:


data_x.drop("summary", axis=1,inplace=True)
data_x.drop('tokens', axis=1,inplace=True)
test.drop("summary", axis=1,inplace=True)
test.drop("tokens", axis=1,inplace=True)


# In[122]:


#Set a copy here, so we can specify the column names for dataframe data_x later
data_x_copy=data_x.copy()
test_copy=test.copy()


# In[123]:


#To sclae the data 
scaler = MinMaxScaler()
scaler.fit(data_x) # this should be the training data
data_x = scaler.transform(data_x)   
test = scaler.transform(test)


# In[124]:


# Create the dataframe
data_x=pd.DataFrame(data_x,columns=data_x_copy.columns)
test=pd.DataFrame(test,columns=test_copy.columns)


# In[125]:


price=train.iloc[:,-1]


# In[126]:


data_y=price
data_y


# ## Model Building

# In[127]:


data_y = np.log(data_y)
data_y.head()


# In[128]:


X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2,random_state=0)


# ## Linear Regression

# In[129]:


# Construc a lineear regression as the benchmark model
#Linear Regression
ols = LinearRegression()
ols.fit(X_train,y_train)

#Lasso
lasso = LassoCV(cv=5)
lasso.fit(X_train,y_train)

#Ratio of Lasso coefficients to OLS coefficients
La_LR=round(np.linalg.norm(lasso.coef_, ord=1)/np.linalg.norm(np.ravel(ols.coef_), ord=1),10)
lasso_pred=lasso.predict(X_test)
rms_lasso_pred = sqrt(mean_squared_error(y_test, lasso_pred))
print('RMSE : {0} '.format(round(rms_lasso_pred,3)))

r2 = r2_score(y_test, lasso_pred)
print(f"r2: {round(r2, 4)}")


# In[135]:


#Ridge
ridge = RidgeCV( cv=5)
ridge_model = ridge.fit(X_train, y_train)

#Ratio of Ridge coefficients to OLS coefficients
Ri_LR = round(np.linalg.norm(ridge.coef_, ord=1)/np.linalg.norm(np.ravel(ols.coef_), ord=1),10)
ridge_pred = ridge.predict(X_test)
rms_ridge_pred = sqrt(mean_squared_error(y_test, ridge_pred))
print('RMSE : {0} '.format(round(rms_ridge_pred,3)))

r2 = r2_score(y_test, ridge_pred)
print(f"r2: {round(r2, 3)}")


# In[136]:


from statlearning import plot_coefficients
plot_coefficients(ridge, X_train.columns)
plt.show()


# In[137]:


ridge_coef = pd.Series(ridge.coef_, index = X_train.columns)
pd.set_option('display.max_rows', 200)
ridge_coef.sort_values()


# In[138]:


print(ridge.get_params)


# In[139]:


print(ridge_model.alpha_)


# In[140]:


#Elastic net 
enet = ElasticNetCV(l1_ratio = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99], cv=5)
enet.fit(X_train, y_train)
enet_pred = enet.predict(X_test)
rms_enet_pred = sqrt(mean_squared_error(y_test, enet_pred))
print('RMSE : {0} '.format(round(rms_enet_pred,3)))

r2 = r2_score(y_test, enet_pred)
print(f"r2: {round(r2, 3)}")


# ## XGboost
#grid search to find the optimised parameters
xgb_model2 = xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0)

tuning_parameters = {
   'learning_rate': [0.01, 0.05, 0.1],
   'n_estimators' : [1000, 1500, 2000],
   'max_depth' : [6,7,8],
   'subsample' : [0.6, 0.8, 1.0]
}

xbst = RandomizedSearchCV(xgb_model2, tuning_parameters, n_iter = 30, cv = 5, random_state=87, n_jobs=4)

xbst.fit(X_train, y_train)
print('Best parameters found by randomised search:', xbst.best_params_, '\n')
# In[141]:


xgb_model2=xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0,subsmaple=0.6,n_estimators=1500,max_depth=6,learning_rate=0.01) 

xgb_model2.fit(X_train.values,y_train)

y_pred_xg2 = xgb_model2.predict(X_test.values)


# In[142]:


rms = sqrt(mean_squared_error(y_test, y_pred_xg2))
#accuracy_xg = xgb_model2.score(X_test.values,y_test)
print('RMSE of XG boosting : {0} '.format(round(rms,3)))

r2 = r2_score(y_test, y_pred_xg2)
print(f"r2: {round(r2, 3)}")


# In[143]:


#from matplotlib import pyplot
#from xgboost import plot_importance
#plot_importance(xgb_model2,max_num_features=20)
#pyplot.show()

from statlearning import plot_feature_importance
plot_feature_importance(xgb_model2, X_train.columns)
plt.title("Feature Importance")
plt.show()


# ## Random Forest 

# In[144]:


param_grid_es = {'n_estimators': [615],
                 'max_depth':[11],
                 'min_samples_split':[6]}

RF_cv = GridSearchCV(ensemble.RandomForestRegressor(random_state=0,verbose=10,oob_score=True),
                     param_grid_es,scoring='neg_mean_squared_error',cv=3)
RF_cv.fit(X_train, y_train)
y_pred_rf = RF_cv.predict(X_test)

accuracy_rf = RF_cv.score(X_test,y_test)
rms_dct_rf = sqrt(mean_squared_error(y_test, y_pred_rf))


# In[145]:


print("Accuracy: {0}" .format(accuracy_rf))
print('RMSE : {0} '.format(round(rms_dct_rf,3)))


# ## bagging

# In[146]:


#due to similarity to random forest, we set n_estimator same to rf 

bag = BaggingRegressor(n_estimators=615, random_state=0,verbose=10)
bag.fit(X_train, y_train)

y_pred_bag=bag.predict(X_test)

rms_dct_bag = sqrt(mean_squared_error(y_test, y_pred_bag))


# In[147]:


print('RMSE : {0} '.format(round(rms_dct_bag,3)))


# ## Regression Tree

# In[148]:


#after grid search with specific tuning in min_samples_leaf and max_depth,
#we find below as reletive good performance
refre_tree = DecisionTreeRegressor(min_samples_leaf=80,
                                   max_depth=15)

#'min_samples_leaf': 46, 'max_depth': 33}
#tuning_parameters = {
#    'min_samples_leaf':np.arange(45,65,1),
#    'max_depth': np.arange(30,60)}
#
#Reg_tree = GridSearchCV(refre_tree, tuning_parameters, cv=3)

refre_tree.fit(X_train, y_train)

regtree_pred = refre_tree.predict(X_test)

rms_regtree_pred = sqrt(mean_squared_error(y_test, regtree_pred))
print('RMSE : {0} '.format(round(rms_regtree_pred,3)))


# ## Stacking

# In[149]:


from sklearn.externals import joblib
from mlxtend.regressor import StackingRegressor
#from sklearn.ensemble import VotingRegressor

import mlxtend

'''
bag; refre_tree; ridge
'''
#clf_xg
models = [RF_cv, xgb_model2,lasso, enet, bag, refre_tree, ridge]

#RF_cv
#xgb_model
#lasso
#RF_cv
#stack = StackingRegressor(models, meta_regressor = lasso, cv=3,n_jobs=-1,verbose=10)
stack = StackingRegressor(models, meta_regressor = lasso,verbose=10)

stack.fit(X_train.values, y_train.values)

#          .ravel())

stack_pred = stack.predict(X_test.values)

rms_stack_pred = sqrt(mean_squared_error(y_test, stack_pred))


# In[150]:


print('RMSE : {0} '.format(round(rms_stack_pred,3)))


# ## Submission

# In[151]:


xgb2_pred_test=xgb_model2.predict(test.values)

xgb2_pred_test=np.exp(xgb2_pred_test)

xgb2_pred_test1=xgb2_pred_test.astype(int)

submission = pd.DataFrame(np.c_[test_csv.index,xgb2_pred_test1],columns=["Id",'price'])
submission.head()


# In[152]:


submission.to_csv("kaggle_xgb2_v11.csv",index=False)


# In[153]:


rf_pred_test=RF_cv.predict(test.values)
rf_pred_test=np.exp(rf_pred_test)

rf_pred_test1=rf_pred_test.astype(int)

submission = pd.DataFrame(np.c_[test_csv.index,rf_pred_test1],columns=["Id",'price'])
submission.head()


# In[154]:


submission.to_csv("kaggle_rfcv_v11.csv",index=False)


# In[155]:


bagging_pred_test=bag.predict(test.values)
bagging_pred_test=np.exp(bagging_pred_test)

bagging_pred_test1=bagging_pred_test.astype(int)

submission = pd.DataFrame(np.c_[test_csv.index,bagging_pred_test1],columns=["Id",'price'])
submission.head()


# In[156]:


submission.to_csv("kaggle_bagging_v11.csv",index=False)


# In[157]:


lasso_pred_test=lasso.predict(test.values)
lasso_pred_test=np.exp(lasso_pred_test)

lasso_pred_test1=lasso_pred_test.astype(int)

submission = pd.DataFrame(np.c_[test_csv.index,lasso_pred_test1],columns=["Id",'price'])
submission.head()


# In[158]:


submission.to_csv("kaggle_lasso_v11.csv",index=False)


# In[159]:


ridge_pred_test=ridge.predict(test.values)
ridge_pred_test=np.exp(ridge_pred_test)

ridge_pred_test1=ridge_pred_test.astype(int)

submission = pd.DataFrame(np.c_[test_csv.index,ridge_pred_test1],columns=["Id",'price'])
submission.head()


# In[160]:


submission.to_csv("kaggle_ridge_v11.csv",index=False)


# In[161]:


enet_pred_test=ridge.predict(test.values)
enet_pred_test=np.exp(enet_pred_test)

enet_pred_test1=enet_pred_test.astype(int)

submission = pd.DataFrame(np.c_[test_csv.index,enet_pred_test1],columns=["Id",'price'])
submission.head()


# In[162]:


submission.to_csv("kaggle_enet_v11.csv",index=False)


# In[163]:


regtr_pred_test=refre_tree.predict(test.values)
regtr_pred_test=np.exp(regtr_pred_test)

regtr_pred_test1=regtr_pred_test.astype(int)

submission = pd.DataFrame(np.c_[test_csv.index,regtr_pred_test1],columns=["Id",'price'])
submission.head()


# In[164]:


submission.to_csv("kaggle_regtr_v11.csv",index=False)


# In[165]:


stack_pred_test=stack.predict(test.values)
stack_pred_test=np.exp(stack_pred_test)
   
stack_pred_test1=stack_pred_test.astype(int)

submission = pd.DataFrame(np.c_[test_csv.index,stack_pred_test1],columns=["Id",'price'])
submission.head()


# In[166]:


submission.to_csv("kaggle_stacking_v11.csv",index=False)


# In[ ]:




