#Goal: In this project we are trying to figure out which
#Audiobook customer is most likely to buy again using Logistic Regression. 
# The data was gathered over 2 years. The targets was data gathered over
#an additional 6 months, seeing who bought or didn't
#buy in that time. 
#%%
#Import The Relevant Libraries
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
#Loading the Raw Data
raw_csv_data= pd.read_csv('Audiobooks-data_raw.csv')
print(display(raw_csv_data.head(20)))
#%%
df=raw_csv_data.copy()
print(display(df.head(20)))
#%%
df=df.drop(['ID'], axis=1)
print(display(df.head()))
#%%
print(df.info())
#%%
#Separate the Targets from the dataset
inputs_all= df.loc[:,'Book length (mins)_overall':'Last visited minus Purchase date']
targets_all= df['Targets']
print(display(inputs_all.head()))
print(display(targets_all.head()))
#%%
#Shuffling the Data to prep for balancing
shuffled_indices= np.arange(inputs_all.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs= inputs_all.iloc[shuffled_indices]
shuffled_targets= targets_all[shuffled_indices]
#%%
#Balance the Dataset
#There are significantly more 0's than 1's in our target.
#We want a good accurate model
print(inputs_all.shape)
print(targets_all.shape)
#%%
num_one_targets= int(np.sum(targets_all))
zero_targets_counter= 0
indices_to_remove= []
print(num_one_targets)
#%%
for i in range(targets_all.shape[0]):
    if targets_all[i]==0:
        zero_targets_counter +=1
        if zero_targets_counter> num_one_targets:
            indices_to_remove.append(i)

#%%

inputs_all_balanced= inputs_all.drop(indices_to_remove,axis=0)
targets_all_balanced= targets_all.drop(indices_to_remove,axis=0)
targets_all_balanced= targets_all_balanced.reset_index(drop=True)
#%%
print(inputs_all_balanced.shape)
print(targets_all_balanced.shape)

#%%
#Standardizing the Inputs
#We won't standardize the Reviews Column because they serve as a 
#categorical variable with a binary 0/1 signifier

print(inputs_all_balanced.columns.values)
#%%
unscaled_inputs= inputs_all_balanced.loc[:,['Book length (mins)_overall', 'Book length (mins)_avg', 'Price_overall',
 'Price_avg', 'Review 10/10', 'Minutes listened', 'Completion',
 'Support requests', 'Last visited minus Purchase date']]

print(display(unscaled_inputs.head()))
#%%
audiobook_scaler= StandardScaler()
#%%
audiobook_scaler.fit(unscaled_inputs)
#%%
scaled_inputs= audiobook_scaler.transform(unscaled_inputs)
#%%
print(display(scaled_inputs))
#%%
print(scaled_inputs.shape)
#%%
#Transforming the scaled_inputs array back into a Dataframe
scaled_inputs= pd.DataFrame(scaled_inputs, columns=['Book length (mins)_overall', 'Book length (mins)_avg',
'Price_overall','Price_avg', 'Review 10/10','Minutes listened', 'Completion','Support requests',
'Last visited minus Purchase date'])
#%%
print(display(scaled_inputs.head()))
#%%
print(display(scaled_inputs.tail(25)))
#%%
#Now adding the 'Review' column to the new scaled inputs Dataframe 
print(inputs_all_balanced['Review'].values)
#%%
review= inputs_all_balanced['Review'].values

#%%
print(review)
#%%
scaled_inputs['Review']= review
#%%
print(display(scaled_inputs.head()))
#%%
print(display(scaled_inputs.tail(25)))
#%%
print(scaled_inputs.info())
#%%
print(scaled_inputs.columns.values)
#%%
reorderd_columns= ['Book length (mins)_overall', 'Book length (mins)_avg', 'Price_overall',
 'Price_avg', 'Review','Review 10/10', 'Minutes listened', 'Completion',
 'Support requests', 'Last visited minus Purchase date']

#%%
scaled_inputs= scaled_inputs[reorderd_columns]
print(display(scaled_inputs.head()))
#%%
#Split the data into Train and Test and Shuffle
#Import relevant library
from sklearn.model_selection import train_test_split
train_test_split(scaled_inputs, targets_all_balanced)

# %%
x_train, x_test, y_train, y_test= train_test_split(scaled_inputs, targets_all_balanced, train_size=0.8, random_state=20)
#%%
#Let's see the shape of these arrays
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %%
#Logistic Regression with SKLearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# %%
#Training the Model
reg= LogisticRegression()
reg.fit(x_train, y_train)

# %%
#Checking the accuracy of the model
reg.score(x_train, y_train)

# %%
#Manually Checking the Model Accuracy
model_outputs= reg.predict(x_train)
print(model_outputs)

# %%
print(model_outputs== y_train)
# %%
np.sum(model_outputs==y_train)

# %%
print(model_outputs.shape[0])

# %%
print(np.sum((model_outputs==y_train))/model_outputs.shape[0])

# %%
#Finding the Intercept/Coefficients and Creating a Summary Table
print(reg.intercept_)

# %%
print(reg.coef_)

# %%
print(scaled_inputs.columns.values)

# %%
feature_name= scaled_inputs.columns.values

# %%
summary_table= pd.DataFrame(columns=['Feature name'], data=feature_name)
# %%
summary_table['Coefficient']= np.transpose(reg.coef_)
print(display(summary_table))

# %%
summary_table.index= summary_table.index+1
summary_table.loc[0]= ['Intercept', reg.intercept_[0]]
summary_table= summary_table.sort_index()
print(display(summary_table))

# %%
#Interpreting the Coefficients
summary_table['Odds_ratio']= np.exp(summary_table.Coefficient)

# %%
print(display(summary_table))

# %%
print(display(summary_table.sort_values('Odds_ratio', ascending=False)))

#%%
#So from the odds ratio table it shows that the
#Avg Book Length, Avg Price, Whether customers left reviews
#or sought customer support were the Strongest indicators of 
#Them Buying again. 
# %%
#Testing the Model
reg.score(x_test, y_test)

# %%
#Getting the Probability of an output
predicted_proba= reg.predict_proba(x_test)
print(predicted_proba)

# %%
print(predicted_proba.shape)

# %%
print(predicted_proba[:,1])
#If the probability is below 0.5 it gets a 0 and above 0.5, it gets a 1
# %%
