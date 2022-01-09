# starbucks_offers

## Datasets

The data is contained in three files:

portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
profile.json - demographic data for each customer
transcript.json - records for transactions, offers received, offers viewed, and offers completed
Here is the schema and explanation of each variable in the files:

portfolio.json

id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) - time for offer to be open, in days
channels (list of strings)
profile.json

age (int) - age of the customer
became_member_on (int) - date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer id
income (float) - customer's income
transcript.json

event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours since start of test. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record

There's an extra df which took long time to populate, called (df_check_all.csv) this is the main df that contains the logic applied in this project

## Logic
I trained an ML model on two datasets:

1- user_offer: Which has unique users as rows, multiple features from the user and offer datasets, and with the user’s most preferred offer as labels

2- offer_user: Which has all offer-user combinations with multiple features, and the label for the data frame is a categorical column with the user’s preferred offer as a value, and if they don’t have an offer preference, we have None as value. I also added another label, which is whether the user viewed and completed the offer or not.

## ML Model
In this project, I used AdaBoostClassifier. 

## Blog Post

I wrote an article that covers all the project, from data cleaning, exploratory analysis, feature engineering to model model building.

article: https://medium.com/@Khalid_OLN/50-off-or-buy-one-get-one-which-one-is-more-effective-1c177d3b9e84
