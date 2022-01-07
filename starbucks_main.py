#!/usr/bin/env python
# coding: utf-8

# In[1]:


# read necessary libraries
import time
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 

# In[2]:


# load datasets
offers = pd.read_json('data/portfolio.json', orient='records', lines=True)
user = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# In[3]:


# take a look at offers df
print(offers.shape)
offers


# In[4]:


# take a look at user df
print(user.shape)
user.head()


# In[5]:


# take a look at transcript df
print(transcript.shape)
transcript.head()


# # Data Cleaning and Feature Engineering
# 
# In this section, I'm gonna clean and create new features for all 3 dfs that will be used in this analysis.
# 
# Below, I'll describe what exactly was cleaned:
# 
# **1- Offer df:**
# 
# - Creating unique name for each offer by using offer type, difficulty, and duration.
# 
# - Unpacking channel column to multiple columns for our ML model
# 
# **2- User df:**
# 
# - Clean age column by changing ages of 118  to nan values (since they're not actually 118)
# 
# - Change gender values  to be numerical: 0 for null, 1 for males, 2 for females, 3 for others
# 
# - Create normalized income column for our ML model, and for analysis
# 
# - Fill income nan values with income columns average
# 
# - Add join datetime column
# 
# - Create user age segment and add new column for it
# 
# - Create columns for join year, month, day
# 
# - group users in cohorts based on their join date
# 
# 
# **3- Transcript df:**
# 
# - unpack value column to multiple columns
# 
# - remove unnecessary columns
# 
# - get some columns from offer df and merge them to the transcript df 
# 
# - conver time from hours to days to match offer duration
# 
# ----
# 
# Finally, I created 4 sub dfs out of transcript df:
# 
# 1- offer received
# 
# 2- offer viewed
# 
# 3- offer completed
# 
# 4- transactions (purchases)
# 
# These will make some of the analysis cleaner

# In[7]:


def offers_cleaning(offers):
    
    '''
    INPUT: df of offers
    
    OUTPUT: df of offers, with added columns: offer_name, channel breakdown, with binary values
    '''
    
    # iterate through al offers
    for index, row in offers.iterrows():
        
        # create a name base on duration, difficulty, and offer type
        offers.loc[index, 'offer_name'] = f"{row['offer_type']}_{row['difficulty']}_{row['duration']}"
    
    # initiate empty columns
    offers[['email', 'web', 'mobile', 'social']] = 0
    
    # iterate through the emty columns
    for index, row in offers.iterrows():
        
        # add binary values, if channel is used then 1
        for i in row['channels']:

            if i == 'email':
                offers.loc[index, 'email'] = 1

            if i == 'web':
                offers.loc[index, 'web'] = 1

            if i == 'mobile':
                offers.loc[index, 'mobile'] = 1

            if i == 'social':
                offers.loc[index, 'social'] = 1
    
    # drop original column
    offers.drop(columns=['channels'], inplace=True)
    
    return offers

def user_cleaning(user):
    
    '''
    INPUT: df of users with gender as strings, and age as numeric.
    
    OUTPUT:
    df of users, with genders switched to numeric (0=N/A, 1=Male, 2=Female, 3=Other.
    age cleaned, adding nan values instead of 118.
    income cleaned and nan values for income are filled with ccolumn average.
    add multiple columns columns for joining date.
    
    '''
    
    
    # clean age values of 118, as they are null values 
    user.loc[user['age'] == 118, ['age']] = np.nan
    
    #update gender to be numeric
    user.loc[user['gender'].isnull(), ['gender']] = 0
    user.loc[user['gender'] == 'M', ['gender']] = 1
    user.loc[user['gender'] == 'F', ['gender']] = 2
    user.loc[user['gender'] == 'O', ['gender']] = 3
    
    # craete normalized income column
    user['income_normalized'] = round((user['income'] - user['income'].min()) / (user['income'].max() - user['income'].min()) * 100)
    
    # fill income NaNs with avg column value
    user['income_normalized'] = user['income_normalized'].fillna(user['income_normalized'].mean())
    user['income'] = user['income'].fillna(user['income'].mean())
    
    # create age_segments columns
    for index, i in enumerate(user['age']):
        
        if i <= 20:
            user.loc[index, 'age_segments'] = '0-20'
            
        elif i <= 40 and i > 20:
            user.loc[index, 'age_segments'] = '20-40'
            
        elif i <= 60 and i > 40:
            user.loc[index, 'age_segments'] = '40-60'
            
        elif i <= 80 and i > 60:
            user.loc[index, 'age_segments'] = '60-80' 
            
        elif i <= 100 and i > 80:
            user.loc[index, 'age_segments'] = '80-100'    

        else:
            user.loc[index, 'age_segments'] = np.nan
    
    # clean join column and seprate it into multiple columns
    user['join_year'] = user['became_member_on'].apply(lambda x: str(x)[:4])
    user['join_month'] = user['became_member_on'].apply(lambda x: str(x)[4:6])
    user['join_day'] = user['became_member_on'].apply(lambda x: str(x)[6:8])
    user['join_date'] = user['became_member_on'].apply(lambda x: f"{str(x)[0:4]}/{str(x)[4:6]}/{str(x)[6:8]}")
    
    # create cohort column
    user['cohort'] = np.nan
    
    # get month name, join it with year and append to cohort column
    for idx, u in user.iterrows():
        user.loc[idx, 'cohort'] = f"{u['join_year']}-{datetime.strptime(u['join_month'], '%m').strftime('%b')}"

    return user


def transcript_cleaning(transcript, offers):
    '''
    INPUT: a transcript df with multiple columns, one of which contains nested values in a dictionary.
    
    OUTPUT:
    cleaned transcript df, unpacking the values in the column with dictionaries to multiple columns.
    convert time from hours to days
    '''
    # unpack value column to multiple columns
    tran_cleaned = transcript['value'].apply(pd.Series)
    tran_cleaned = transcript.join(tran_cleaned)
    
    #megre offer id columns
    tran_cleaned['offer id'].fillna(tran_cleaned['offer_id'], inplace=True)
    tran_cleaned.drop(columns=['offer_id', 'value'], inplace=True)
    tran_cleaned['offer_id'] = tran_cleaned['offer id']
    tran_cleaned.drop(columns=['offer id'], inplace=True)
    
    #merge columns from offer metadata to transaction table
    offer_type = offers[['id','offer_type', 'duration', 'difficulty']]
    tran_cleaned = tran_cleaned.merge(offer_type, left_on='offer_id', right_on='id', how='left').drop(columns=['id'])
    
    #change hours to days
    tran_cleaned['time'] = tran_cleaned['time'] / 24
    
    return tran_cleaned


# In[8]:


# run cleaning functions and store theiir returns

tran_cleaned = transcript_cleaning(transcript, offers)
user = user_cleaning(user)
offers = offers_cleaning(offers)


# In[9]:


def create_sub_dfs(df):
    
    '''
    
    INPUT: transcript df wich contains all events
    
    OUTPUT: 4 seperate dfs created from unique events from the transcript df
    
    '''
    
    df_dict = {}
    
    # iterate through unqiue events
    for index, event in enumerate(tran_cleaned['event'].unique()):
        
        #create a sub df filtering only the index event and store in a dict
        df_dict[event] = tran_cleaned[tran_cleaned['event'] == event]
        df_dict[event] = df_dict[event].reset_index().drop(columns=['index', 'event'])
        
    return df_dict

# store the dictionary
df_dict = create_sub_dfs(tran_cleaned)


def clean_sub_dfs(df_dict):
    
    '''
    
    INPUT: dictionary of dfs
    OUTPUT: 4 sperate dfs, each one with 1 unique event data. removed columns that have 100% empty values
    
    '''
    
    #iterate through dfs
    for sub_df in df_dict:
        
        # iterate through df columns
        for column in df_dict[sub_df]:
            
            # removee column if it's 100% nans
            if df_dict[sub_df][column].isna().mean() > 0:
                df_dict[sub_df].drop(columns=[column], inplace=True)
    
    return df_dict['offer received'], df_dict['offer viewed'], df_dict['transaction'], df_dict['offer completed']
    


# In[10]:


#create sub dfs for each event type
df_received, df_viewed, df_transaction, df_completed = clean_sub_dfs(df_dict)

df_received.shape, df_viewed.shape, df_transaction.shape, df_completed.shape


# # Exploratory Analysis
# 
# In this section we want to analyis all dfs and try to understand on high level, some stats and grapgh that wil describe our data better.

# ### Checking all dfs

# In[11]:


user.head()


# In[12]:


tran_cleaned.head()


# ### Printing our user df description

# In[14]:


user.describe()


# ### plotting basic columns from our user df

# In[15]:


user[['age', 'income_normalized']].hist();


# In[16]:


user['age_segments'].hist();


# ### Printing some basic stats
# 
# We want to look are some rates that will help us understand the defference of effect each offer provides

# In[17]:


def get_basic_stats(received_count, viewed_count, completed_count):
    
    print("Offers Received:",received_count)
    print("Offers Viewed:",viewed_count)
    print("Offers Completed:",completed_count)
    print()
    print("View Through Rate:", int(viewed_count / received_count * 100),"%")
    print("Completion Rate:", int(completed_count / received_count * 100),"%")
    print()


# #### looking at all offers combined

# In[18]:


received_count_all = len(df_received)
viewed_count_all = len(df_viewed)
completed_count_all = len(df_completed)

get_basic_stats(received_count_all, viewed_count_all, completed_count_all)


# #### looking at bogo offers

# In[19]:


received_count_bogo = len(df_received[df_received['offer_type'] == 'bogo'])
viewed_count_bogo = len(df_viewed[df_viewed['offer_type'] == 'bogo'])
completed_count_bogo = len(df_completed[df_completed['offer_type'] == 'bogo'])

get_basic_stats(received_count_bogo, viewed_count_bogo, completed_count_bogo)


# #### looking at discout offers

# In[20]:


received_count_bogo = len(df_received[df_received['offer_type'] == 'discount'])
viewed_count_bogo = len(df_viewed[df_viewed['offer_type'] == 'discount'])
completed_count_bogo = len(df_completed[df_completed['offer_type'] == 'discount'])

get_basic_stats(received_count_bogo, viewed_count_bogo, completed_count_bogo)


# #### looking at informational offers

# In[21]:


received_count_bogo = len(df_received[df_received['offer_type'] == 'informational'])
viewed_count_bogo = len(df_viewed[df_viewed['offer_type'] == 'informational'])
completed_count_bogo = len(df_completed[df_completed['offer_type'] == 'informational'])

get_basic_stats(received_count_bogo, viewed_count_bogo, completed_count_bogo)


# #### caluculating total received rewards by offer type

# In[22]:


# calculate rewards received by offer type
df_completed.groupby("offer_type")['reward'].sum().reset_index()


# ### Now, let's look at some graphs!

# here, we will take a look at daily purchases as a line graph, and then plotting red dots to show when our offers were sent by day. This is to illustrate offers effect on purchases

# In[23]:


# aggregating necessary data
offer_sent_times = df_received['time'].unique()
amounts_by_time = df_transaction.groupby('time')['amount'].count().reset_index()

# store 1 x and 2 ys
x = amounts_by_time['time']
y_1 = amounts_by_time['amount']
y_2 = offer_sent_times


# aligning y_2 shape with x
zeros = np.zeros(120)

for index, i in enumerate(x):
    for idx, y in enumerate(y_2):
        if i == y:
            # random number to show the dots properly
            if y == 0:
                zeros[index] = y+2
            else:
                zeros[index] = y

# store y_2 after alignment
y_2 = zeros

#plotting a line chart
fig, ax1 = plt.subplots()
ax1.plot(x, y_1);
plt.xlim([-1, 30])
plt.xlabel('Time', size=10)
plt.ylabel('Transactions Count', size=10)

ax2 = ax1.twinx()
ax2.scatter(x, y_2, color='red');
plt.ylabel('Offers Sent Day', size=10)
plt.ylim([1, 30]);


# #### Let's look at our daily revenue from the transaction table

# In[24]:


# Plotting Revenue by day
rev_x = df_transaction.groupby('time').sum()['amount'].keys()
rev_y = df_transaction.groupby('time').sum()['amount'].values

plt.bar(rev_x, rev_y, width=0.3, color='green')
plt.xlabel('days', size=10)
plt.ylabel('revenue', size=10)
plt.show()


# #### Here, let's look at offer type as a bar graph for each gender

# In[25]:


# Merge transactions with user table
df_completed_user = pd.merge(df_completed, user, left_on='person', right_on='id').drop(columns=['id', 'income_normalized', 'became_member_on'])


# create seperate counts for each gender type
male = list(df_completed_user.loc[df_completed_user['gender'] == 1].groupby('offer_type').count()['gender'])
female = list(df_completed_user.loc[df_completed_user['gender'] == 2].groupby('offer_type').count()['gender'])
other = list(df_completed_user.loc[df_completed_user['gender'] == 3].groupby('offer_type').count()['gender'])

# create labels for the graph
labels = ['bogo', 'discount']
width = 0.4


# plotting
fig, ax = plt.subplots()

ax.bar(labels, other, width, label='other', color='red')
ax.bar(labels, male, width, label='male', bottom=other, color='gold')
ax.bar(labels, female, width, label='female', bottom=male, color='orange')


ax.set_ylim([0, 15000])
ax.set_ylabel('Completed Offers')
ax.set_title('Offers Completed By Type and Gender')

plt.xticks(labels)
plt.yticks(np.arange(0, 20000, 2000))

ax.legend()
plt.show()


# #### Let look at age range offer prefereences

# In[26]:


# create seperate counts for each age range
age_0_20 =  df_completed_user.loc[df_completed_user['age_segments'] == '0-20'].groupby('offer_type').count()['age_segments']
age_20_40 =  df_completed_user.loc[df_completed_user['age_segments'] == '20-40'].groupby('offer_type').count()['age_segments']
age_40_60 =  df_completed_user.loc[df_completed_user['age_segments'] == '40-60'].groupby('offer_type').count()['age_segments']
age_60_80 =  df_completed_user.loc[df_completed_user['age_segments'] == '60-80'].groupby('offer_type').count()['age_segments']
age_80_100 =  df_completed_user.loc[df_completed_user['age_segments'] == '80-100'].groupby('offer_type').count()['age_segments']

# create labels for the graph
labels = ['bogo', 'discount']
width = 0.4


# plotting
fig, ax = plt.subplots()

ax.bar(labels, age_0_20, width, label='0-20', color='red')
ax.bar(labels, age_80_100, width, label='80-100', bottom=age_0_20, color='brown')
ax.bar(labels, age_20_40, width, label='20-40', bottom=age_80_100, color='gold')
ax.bar(labels, age_40_60, width, label='40-60', bottom=age_20_40, color='orange')
ax.bar(labels, age_60_80, width, label='60-80', bottom=age_40_60, color='purple')



ax.set_ylim([0, 15000])
ax.set_ylabel('Completed Offers')
ax.set_title('Offers Completed By Type and Age Range')

plt.xticks(labels)
plt.yticks(np.arange(0, 20000, 2000))

ax.legend()
plt.show()


# #### Let's take a look at cohort data.
# 
# This graph shows users who completed offers grouped by their joining year

# In[83]:


# store x and y for completed offers by joining year
x_cohort = df_completed_user.groupby('join_year').count()['person'].reset_index().sort_values(by=['join_year'])['join_year']
y_cohort = df_completed_user.groupby('join_year').count()['person'].reset_index().sort_values(by=['join_year'])['person']

plt.bar(x_cohort, y_cohort);


# In[96]:


plt.bar(user.groupby('join_year').count()['id'].keys(), user.groupby('join_year').count()['id'].values, color='green');


# #### Revenue by offer
# 
# Since we don't have offer id for purchases, it would be faster to sum the difficulty column for completed offers to get how much did people pay for each offer in total.

# In terms of revenue, it seems that discout  offers bring moree revenue that bogo offers, you can sse that 4 out of top 5 offers are discounts. 

# In[28]:


rev_by_offer = tran_cleaned.merge(offers, left_on='offer_id', right_on='id', how='left')
rev_by_offer = rev_by_offer[['person', 'event', 'time', 'amount', 'reward_x', 'offer_id', 'offer_name','offer_type_x', 'duration_x', 'difficulty_x']]
rev_by_offer = rev_by_offer[rev_by_offer['event'] == 'offer completed']
rev_by_offer = rev_by_offer.groupby('offer_name').sum().reset_index().sort_values(by='difficulty_x',ascending=False)
#rev_by_offer.columns = ['offer_name', 'revenue']

plt.barh(rev_by_offer['offer_name'][:8], rev_by_offer['difficulty_x'][:8], color='purple')
plt.xlabel('Revenue')
plt.ylabel('Offer Name')
plt.show()


# In[97]:


rev_by_offer = tran_cleaned.merge(offers, left_on='offer_id', right_on='id', how='left')
rev_by_offer = rev_by_offer[['person', 'event', 'time', 'amount', 'reward_x', 'offer_id', 'offer_name','offer_type_x', 'duration_x', 'difficulty_x']]
rev_by_offer = rev_by_offer[rev_by_offer['event'] == 'offer completed']
rev_by_offer = rev_by_offer.groupby('offer_name').count().reset_index().sort_values(by='difficulty_x',ascending=False)
#rev_by_offer.columns = ['offer_name', 'revenue']

plt.barh(rev_by_offer['offer_name'][:8], rev_by_offer['difficulty_x'][:8], color='gold')
plt.xlabel('Revenue')
plt.ylabel('Offer Name')
plt.show()


# # Further Feature Engineering
# 
# This is the most important step.
# Here, we will create a new binary column for each user-offer combination and check if the offer affected the user to make a purchase and complete the offer or not.
# Basically, we will check whether the user received, viewed and completed that offer in the given offer duration.
# For users who didn't view that offer, we will not consider the offer effective with them.
# And for the informational offers, since they don't have completion event, we need to account for that, by checking if the user received and viewed the offer and within the informational offer duration made a purchase.

# In[29]:


def user_offer_df(user):
    
    '''
    
    INPUT: user id
    
    OUTPUT: a user df with offer ids as rows and binary values and event types as columns ['received', 'viewed', 'completed']
    
    '''    
    
    # initiate empty df
    df_user = pd.DataFrame(columns=['offer_id', 'received', 'viewed', 'completed'])
    
    # iterate through of ids
    for index, offer in enumerate(offers['id']):
        
        # store offer id as the main df column
        df_user.loc[index, 'offer_id'] = offer
        
        # if a user recieved this offer id iteration, add 1 as value
        if len(df_received[(df_received['person'] == user) & (df_received['offer_id'] == offer)]) > 0:
            df_user.loc[index, 'received'] = 1
        else:
            df_user.loc[index, 'received'] = 0
            
        # if a user viewed this offer id iteration, add 1 as value
        if len(df_viewed[(df_viewed['person'] == user) & (df_viewed['offer_id'] == offer)]) > 0:
            df_user.loc[index, 'viewed'] = 1
        else:
            df_user.loc[index, 'viewed'] = 0
            
        # if a user completed this offer id iteration, add 1 as value    
        if len(df_completed[(df_completed['person'] == user) & (df_completed['offer_id'] == offer)]) > 0:
            df_user.loc[index, 'completed'] = 1
        else:
            df_user.loc[index, 'completed'] = 0
        
        
    return df_user



def user_update_info(user):

    '''
    
    INPUT: user id
    
    OUTPUT: creates a user df using (user_offer_df).
    checks if a user made a purchase within the informational offer duration, then updates informational offers (completed) column 
    
    '''    
    
    # create user df using the above function
    df_user = user_offer_df(user)
    
    # create a df with received informatioal offers only for a particular user
    info_table = df_received[(df_received['person'] == user) & (df_received['offer_type'] == 'informational')]
    
    # create a df with user's purchases
    purchase_table = df_transaction[df_transaction['person'] == user]
    
    
    # if a user received an informational offer
    if len(info_table) > 0:
        
        # iterate through informational offers
        for index, row in info_table.iterrows():
            
            # store duration time and when was the offer received
            duration = row['duration']
            received_time = row['time']
            
            # iterate through the purchase table to see if there's any purchases with in the duration period
            for idx, r in purchase_table.iterrows():
                
                purchase_time = r['time']
                
                if purchase_time-received_time <= duration:
                    
                    df_user.loc[df_user['offer_id'] == row['offer_id'], 'completed'] = 1
                    
                else:
                    pass
                
    return df_user



def create_combined_df(num_users=10):

    '''
    
    INPUT: number of users
    
    OUTPUT: iterates through the number of users inputted and create a df with each user and promotion combanation.
    and wether the user was effected by the offer or not.
    Creates the following columns [user_id, offer_id, offer_effect, offer_completed]
    
    '''    
    # initiate empty combined df       
    df = pd.DataFrame(columns=['user_id', 'offer_id','offer_effect', 'offer_completed'])
    
    # index for iteration
    idx = 0
    counter = 0
    
    # function run time
    begin = time.time()
    
    #iterate through users
    for i in user['id'][:num_users]:
        
        # create a user df
        user_df = user_update_info(i)

        # iterate through the user df
        for index, row in user_df.iterrows():

            # store user id in the combined df
            df.loc[idx, 'user_id'] = i
            
            # store offer id in the combined df
            df.loc[idx, 'offer_id'] = row['offer_id']
            
            # check if offer is completed
            if row['completed'] == 1:
                df.loc[idx, 'offer_completed'] = 1
            if row['completed'] == 0:
                df.loc[idx, 'offer_completed'] = 0
            
            # check if offer is completed after viewing it
            if row['received'] + row['viewed'] + row['completed'] == 3:
                df.loc[idx, 'offer_effect'] = 1
            
            # check if offer was received
            if row['received'] + row['viewed'] + row['completed'] == 0:
                df.loc[idx, 'offer_effect'] = np.nan
            
            # check if offer was recived and completed but not viewed
            if row['received'] + row['viewed'] + row['completed'] == 1:
                df.loc[idx, 'offer_effect'] = 0
            if row['received'] + row['viewed'] + row['completed'] == 2:
                df.loc[idx, 'offer_effect'] = 0

            idx += 1
        counter += 1
        
        if counter % 500  == 0:
            print(counter)
            
    end = time.time()
    
    print(end - begin)

    return df


# # Running the functions!
# 
# The cell below is to run the functions above, and since we have 17K users and +300K transaction, it takes a couple of hours to run. 
# 
# To save your time, I already ran the functions on all users and transactions/events and stored the df in a csv file called 'df_check_all.csv'.
# 
# in the second cell, I wrote down a line to read the csv file to make it faster

# In[30]:


#df_check = create_combined_df(17000)
#df_check.to_csv('df_check_all.csv')


# In[31]:


df_check = pd.read_csv('df_check_all.csv').drop(columns=['Unnamed: 0'])


# In[32]:


df_check.shape


# # Creating Final Datasets for ML Model
# 
# In this stage, I'm creating two different data sets out of df_check:
# 
# 1- ***user_offer:*** 
# Which has unique users as rows, multiple features from user and offer datasets, and with the user most preferred offer as labels 
# 
# 2- ***offer_user:*** 
# Which has all offer-user combanations with mulltiple features, and wether the user preferes the offer or not as labels
# 
# 
# 
# 

# In[69]:


def create_user_offer_df(df):
    
    
    '''
    
    INPUT: takes df with user offer combinations.
    
    OUTPUT: returns a df with unique users as rows and and if they prefer offers, which type of offer is their best offer.
    
    '''    
    
    df_offer_merged = df.merge(offers, left_on='offer_id', right_on='id', how='left')[['user_id', 'offer_type', 'offer_effect']]
    
    unstack_df = df_offer_merged.groupby(['user_id', 'offer_type'])['offer_effect'].sum().unstack()
    
    df_matrix = unstack_df.merge(user, left_on='user_id', right_on='id')[['id','age','gender', 'income', 'income_normalized', 'join_year', 'join_month', 'join_day','bogo', 'discount', 'informational']]
    
    df_matrix = df_matrix.dropna()
    
    for index, row in df_matrix.iterrows():
        
        if row[['bogo', 'discount', 'informational']].sum() == 0:
            
            df_matrix.loc[index, 'final'] = 'None'
            
        else:
            
            df_matrix.loc[index, 'final'] = df_matrix[['bogo', 'discount', 'informational']].idxmax(axis=1)[index] 
            
    df_matrix = df_matrix.drop(columns=['id', 'bogo', 'discount', 'informational'])
    
    return df_matrix

def create_offer_user_df(df, second_label=False):

    
    '''
    
    INPUT: takes a df with user offer combination.
    
    OUTPUT: returns a df with every user offer combination and wether the user was interested in the offer or not.
    
    '''    
    #if second_label == False:
    #    df = df[~df['offer_effect'].isna()]

    
    df = df.merge(user, left_on='user_id', right_on='id')
    
    df = df.merge(offers, left_on='offer_id', right_on='id')
    
    if second_label == False:
        df = df[['age', 'became_member_on', 'gender', 'income_normalized','income', 'join_year', 'join_month', 'join_day', 'email', 'web', 'social', 'mobile', 'duration', 'difficulty', 'offer_type', 'offer_effect']]
    else:
        df = df[['age', 'became_member_on', 'gender', 'income_normalized','income', 'join_year', 'join_month', 'join_day', 'email', 'web', 'social', 'mobile', 'duration', 'difficulty', 'offer_type', 'offer_effect','offer_completed']]
    
    df = df.dropna()

    for index, row in df.iterrows():
        if row['offer_effect'] == 0:
            df.loc[index, 'offer_effect'] = 'None'
        else:
            df.loc[index, 'offer_effect'] = df.loc[index, 'offer_type']
            
    df = df.drop(columns=['offer_type'])
    
    return df


# In[71]:


# user_offer: Which has unique users as rows, multiple features from user and offer datasets,
# and with the user most preferred offer as labels

user_offer = create_user_offer_df(df_check)

# offer_user: Which has all offer-user combanations with mulltiple features,
# and wether the user preferes the offer or not as labels

offer_user = create_offer_user_df(df_check, second_label=True)


# ### Let's take a look at user offer table

# In[72]:


print(user_offer.shape)
user_offer.head()


# ### Now, let's take a look at offer user table

# In[73]:


print(offer_user.shape)
offer_user.head()


# # Model Building, Tuning and Scoring
# 
# 
# I wrote two functions to help me test out multiple variation and to tune in my model.
# I tested multiple ML algorithims and got varied accuracy scores for each one, without using GridSearchCV, I got the following accuracy scores for these ML models.
# 
# GaussianNB = **0.45**
# 
# LogisticRegression = **0.45**
# 
# SVC = **0.45**
# 
# DecisionTreeClassifier = **0.44**
# 
# KNeighborsClassifier = **0.38**
# 
# AdaBoostClassifier = **0.48**
# 
# Then I tried adding more and more features: channel breakdown with binary values, filling NaN income values with avg column values, adding normalized income column, creating two different datasets, one with unique users and their preferred offer, and another dataset with each offer-user possible combanations and wether the user preferes this type of offer or not.
# 
# 
# I did not see any significant improvement in teh accuracy, or F1 scores.
# Then I implemented GridSearchCV with multiple n_estimators and multiple learning_rates

# #### Store 4 different ML models to find teh best fit

# In[37]:


model_SVC = SVC()
model_DTC = DecisionTreeClassifier()
model_GNB = GaussianNB()
model_ABC = AdaBoostClassifier()


# #### Store params for GridSearchCV

# In[38]:


parameters = {'n_estimators': [100, 500, 1000, 1500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2]}


# #### Creating functions to train and evaluate our models

# In[55]:


def train_model(df, features, labels, model):
    
    '''
    
    INPUT: a df with it's features and labels, a selected ML model, and set of params for GridSearchCV
    
    OUTPUT: a trained model, X test set, y test set
    
    '''    

    # create features and labels sets
    X = df[features]
    y = df[labels]
    
    # create train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=43)
    
    #store GridSearchCV model
    #model = GridSearchCV(model, params)
    
    # train the model
    model = model.fit(X_train, y_train)
    
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    
    '''
    
    INPUT: takes a trained model, X test set, y test set
    
    OUTPUT: prediction accuracy, classification report
    
    '''    
    
    # store predicted y test labels
    pred = model.predict(X_test)
    
    # get accuracy score
    acc_score = accuracy_score(y_test,pred)
    
    # get classification report
    class_report = classification_report(y_test,pred)
    
    print("prediction accuracy score is: {}".format(acc_score))
    print("classification report:")
    print(class_report)


# #### first, let's train and evaluate our model  on the user offer dataset

# In[74]:


#user_offer: age, gender, income, join dates as features
model, X_test, y_test = train_model(user_offer, ['age', 'gender', 'income', 'join_year', 'join_month', 'join_day'], 'final', model_ABC)
evaluate_model(model, X_test, y_test)


# #### Now, let's train and evaluate our model  on the offer user dataset with just age, gender and income as features.

# In[75]:


#offer_user
model, X_test, y_test = train_model(offer_user, ['age', 'gender', 'income'], 'offer_effect', model_ABC)
evaluate_model(model, X_test, y_test)


# #### Finally, let's train and evaluate our model on the offer user dataset with more features:
# 
# 
# ['age', 'gender', 'income_normalized','email', 'web', 'social', 'mobile', 'join_year', 'join_month', 'join_day', 'difficulty', 'duration']

# In[76]:


#offer_user with more features 
model, X_test, y_test = train_model(offer_user, ['age', 'gender', 'income_normalized','email', 'web', 'social', 'mobile', 'join_year', 'join_month', 'join_day', 'difficulty', 'duration'], 'offer_effect', model_ABC)
evaluate_model(model, X_test, y_test)


# In[77]:


#offer_user with compleeted offer as labels 
model, X_test, y_test = train_model(offer_user, ['age', 'gender', 'income_normalized','email', 'web', 'social', 'mobile', 'join_year', 'join_month', 'join_day', 'difficulty', 'duration'], 'offer_completed', model_ABC)
evaluate_model(model, X_test, y_test)


# In[1295]:


### further notes and improvments


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




