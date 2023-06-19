#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
#import swifter
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv('diabetic_data.csv')
df.shape


# In[5]:


import swifter


# In[6]:


df1=pd.DataFrame(df.isna().sum())
df1=df1.reset_index()
df1.columns=['Column_Names','Count_of_Nan_Values']
df2=df1[df1['Count_of_Nan_Values']!=0].sort_values(by=['Count_of_Nan_Values'],ascending=False)
df2['Percentage_of_NAN']=df2['Count_of_Nan_Values']/len(df)*100
print('The Nan Value columns with percentage are as follows')
print(df2)


# In[7]:


df.head(10)


# # Replacing Question Mark with NAN

# In[8]:


import numpy as np
df.replace({'?':np.nan},inplace=True)
df1=pd.DataFrame(df.isna().sum())
df1=df1.reset_index()
df1.columns=['Column_Names','Count_of_Nan_Values']
df2=df1[df1['Count_of_Nan_Values']!=0].sort_values(by=['Count_of_Nan_Values'],ascending=False)
df2['Percentage_of_NAN']=df2['Count_of_Nan_Values']/len(df)*100
print('The Nan Value columns with percentage are as follows')
print(df2)


# # EDA and Data Cleaning

# In[9]:


#Dropping columns with more than 40 percent null values
df.drop(['weight','payer_code','medical_specialty'],axis=1,inplace=True)
#Changing the readmitted column
df['readmitted'] = df['readmitted'].replace({'>30':1,'<30':1,'NO':0})
#Replacing Age with mean
df['age'] = df['age'].replace({'[70-80)': 75, '[60-70)': 65, '[50-60)': 55, '[80-90)': 85, '[40-50)': 45, '[30-40)': 35, '[90-100)': 95, '[20-30)': 25, '[10-20)': 15, '[0-10)': 5})


# In[10]:


sns.histplot(df['age'],kde=True,bins=10)
plt.title('Histogram with KDE of Age vs Frequency')
plt.xticks(rotation=90);


# In[11]:


sns.histplot(df['age'],kde=True,bins=10)
plt.xticks(rotation=90);


# # Checking the distribution of Age

# In[19]:


from scipy.stats import shapiro
#The Shapiro-Wilk test (often referred to as the Shapiro test) 
#is a statistical test used to assess whether a set of data follows a normal distribution
stat,p = shapiro(df['age'])
alpha = 0.05
#Checking the signiface value
if p > alpha:
    #The Null hypothesis is a distribution sample is Normally distributed
    print('Sample seems Gaussian (fail to reject H0)')
    #The Alternate hypothesis is a distribution is not normally distributed
else:
    print('Sample does not seems Gaussian (reject H0)')


# In[12]:


from scipy.stats import chisquare
from numpy.random import poisson
chi_statistic, p_value = chisquare(df['age'])
alpha = 0.05
if p_value > alpha:
    print('Sample follows a Poisson distribution (fail to reject H0)')
else:
    print('Sample does not follow a Poisson distribution (reject H0)')


# In[13]:


from scipy.stats import skewtest
#The skewtest is a statistical test that is used to assess
#whether a set of data is symmetric or skewed.
#The test is based on the comparison of the observed skewness 
#of data to the expected skewness under the assumption of normality. 
statistic, p_value = skewtest(df['age'])
alpha = 0.05
#Checking the significance value
if p_value > alpha:
    #Null hypothesis is data is not skewed
   print('Sample is symmetric (fail to reject H0)')
    #Alternate hypothesis is data is skewed
else:
   print('Sample is skewed (reject H0)')


# In[14]:


sns.histplot(df['num_lab_procedures'],kde=True);
plt.title('Number_of_lab_procedures Histogram')
plt.show()


# In[15]:


from scipy.stats import shapiro
stat, p = shapiro(df['num_lab_procedures'])
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[16]:


from scipy.stats import skewtest
statistic, p_value = skewtest(df['num_lab_procedures'])
alpha = 0.05
if p_value > alpha:
   print('Sample is symmetric (fail to reject H0)')
else:
   print('Sample is skewed (reject H0)')


# In[17]:


sns.histplot(df['num_medications'],kde=True)
plt.title('Number of medications histogram')
plt.show()


# In[18]:


from scipy.stats import skewtest
statistic, p_value = skewtest(df['num_medications'])
alpha = 0.05
if p_value > alpha:
   print('Sample is symmetric (fail to reject H0)')
else:
   print('Sample is skewed (reject H0)')


# In[19]:


import numpy as np
import scipy.stats as stats
skewness = stats.skew(df['num_medications'])
print("Skewness:", skewness)
if skewness > 0:
    print("The distribution is normal distribution with right-tailed.")
else:
    print("The distribution is not right-tailed.")


# In[20]:


sns.scatterplot(data=df, y='num_medications', x='time_in_hospital', hue='readmitted');
plt.title('Scatterplot of Num_Medications and Time_In_Hospital')
plt.show()


# In[21]:


sns.barplot(data=df, x='time_in_hospital', y='num_medications', hue='readmitted');


# In[22]:


sns.scatterplot(data=df, x='num_medications', y='num_lab_procedures', hue='readmitted')
plt.title('Scatterplot of num_lab_procedures and num_medications')
plt.show()


# In[23]:


df.corr()[['num_lab_procedures','num_medications']].loc[['time_in_hospital','num_lab_procedures']]


# In the first scatter plot, we can observe that there is a weak positive relationship between time_in_hospital and num_medications. We can also see that most of the data points are concentrated in the lower range of time_in_hospital and num_medications. Additionally, we can see that the readmitted variable does not seem to have a strong relationship with either variable.
# 
# In the second scatter plot, we can observe that there is a weak positive relationship between num_medications and num_lab_procedures. We can also see that most of the data points are concentrated in the lower range of num_medications and num_lab_procedures. Additionally, we can see that the readmitted variable does not seem to have a strong relationship with either variable.
# 
# Overall, the scatter plots suggest that there is a weak positive relationship between the variables, but it is not a strong relationship. Additionally, the readmitted variable does not seem to be strongly related to either variable. However, further analysis and statistical tests may be necessary to confirm these observations.

# # Feature Selection before Applying K Means ALgorithm

# In[24]:


df.columns


# In[33]:


df_diabetes=df.copy()


# # Eliminating patient_nbr and encounter_id

# In[34]:


#Eliminating encounter_id and patient_nbr
#Since K means clustering works on similarity or difference and the objective of K means is to cluster records hence
#Presence of a unique key will be meaningless and hence it wont contribute to the clustering
#Remove Id variables as they dont contribute to clustering
df_diabetes.drop(columns=['encounter_id','patient_nbr'],axis=1,inplace=True)


# # Checking categorical and Continous Features

# In[35]:


num_features = df_diabetes.select_dtypes(include=['int', 'float'])
df_diabetes.head(10)


# In[36]:


cat_features = df_diabetes.select_dtypes(include=['object']).columns
print(cat_features)


# # Taking Value counts on dataset

# In[37]:


for column in cat_features:
    #Taking Value counts for each categorical column
    value_counts = df_diabetes[column].value_counts()
    prop_percentage=value_counts/len(df_diabetes) *100
    #Setting threshold to 95 percentage, So if a class in a column has more than 95 percent values we
    imbalanced_values=prop_percentage>95
    if not imbalanced_values.empty:
        print(f"Column {column} has imbalanced classes:")
        print(prop_percentage[imbalanced_values])


# # Reasons for dropping Imbalanced data
# If a column in a dataset has the same value for all data points,it will not be useful for clustering. 
# This is because clustering algorithms rely on differences or similarities between data points to group them into clusters.
# When a column has the same value for greater than 95 percent of datapoints, There is less variability in those datapoints which will introduce Skewness in the clustering process.
# Additionally,it means that this column provides no useful information for distinguishing between data points.
# Since K-Means is a distance based clustering algorithm,columns having less variability will dominate the results and hence other seemingly important columns will have less impact.
# 
# 

# In[38]:


imbalanced_data=['examide','metformin-rosiglitazone','metformin-pioglitazone','glimepiride-pioglitazone','glipizide-metformin','glyburide-metformin','citoglipton','tolazamide','troglitazone','miglitol','acarbose','tolbutamide','acetohexamide','chlorpropamide','nateglinide','repaglinide']
df_diabetes.drop(columns=imbalanced_data,inplace=True)


# # Handling other categorical variables
# Since the other columns have a particular order in the dosage/Value we can perform label encoding to these columns, An example is shown by the value counts of Max_glu_serum Column
# None    96420
# Norm     2597
# greater than 200     1485
# greater than 300     1264
# Hence we can do label encoding and this wont affect the distance meteric since the values are determined by types as mentioned above with None being mapped to 0

# In[39]:


#Label Encoding in data where there is a ordinality
from sklearn.preprocessing import LabelEncoder
ordinal_columns=['max_glu_serum', 'A1Cresult',
       'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
       'rosiglitazone', 'insulin', 'change', 'diabetesMed']
df_diabetes[['max_glu_serum', 'A1Cresult',
       'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
       'rosiglitazone', 'insulin', 'change', 'diabetesMed']] = df_diabetes[['max_glu_serum', 'A1Cresult',
       'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
       'rosiglitazone', 'insulin', 'change', 'diabetesMed']].swifter.apply(LabelEncoder().fit_transform)


# # Working on other columns 

# In[40]:


#Working on Data Classification for remaining columns
df_diabetes = df_diabetes.drop(df_diabetes.loc[df_diabetes["gender"]=="Unknown/Invalid"].index, axis=0)


# In[41]:


ordinal_columns=['gender','race']
one_hot = pd.get_dummies(df_diabetes[['gender','race']])
df_diabetes=pd.concat([df_diabetes,one_hot],axis=1)


# # Reasons for Eliminating three columns diag_1,diag_2,diag_3
# The diag_1, diag_2, and diag_3 columns in the Diabetes dataset contain the ICD-9 codes for the primary, secondary, and additional diagnoses of the patients. Each ICD-9 code corresponds to a specific medical condition or diagnosis, and there are thousands of possible codes.
# Including these columns in the clustering analysis can result in a high-dimensional dataset with a large number of unique values, which can make it difficult to identify meaningful clusters or interpret the results. Moreover, the presence of these columns can lead to overfitting, as the clustering algorithm may focus too much on these columns and generate clusters based on ICD-9 codes rather than underling patterns in the data

# In[42]:


df_diabetes.drop(columns=['diag_1','diag_2','diag_3','gender','race'],inplace=True)


# In[43]:


df_diabetes_final=df_diabetes.copy()


# In[255]:


df_diabetes_final.shape


# # K-MEANS Using Tou 

# In[441]:


import numpy as np
import swifter
from scipy.spatial.distance import euclidean

         
def get_random_centroids(input_dataframe,no_of_clusters):
    '''
    The function takes a dataframe as an input and creates a random K centroids from uniform distribution
    '''
    #Initialize random centroids from dataset
    list_of_centroids = []
    
    for cluster in range(no_of_clusters):
        #Generates a centroids randomly from uniform distribution 
        random_centroid = input_dataframe.swifter.apply(lambda x:float(x.sample()))
        #From the given dataset it randomly selects centroids
        list_of_centroids.append(random_centroid)
    
    centroid_df=pd.concat(list_of_centroids,axis=1)
    #Naming the column as Label for ease of purpose
    centroid_df.index.name='Cluster_Assigned'
    '''
    The function returns a dataframe consisting of no of clusters required
    '''
    return centroid_df

def get_labels(input_dataframe,centroid_df):
    '''
    This function takes centroids as input and takes the initial dataframe and gives them labels to which cluster
    they belong to
    '''
    euclidean_distances = centroid_df.swifter.apply(lambda x: np.sqrt(((input_dataframe - x) ** 2).sum(axis=1)))
    #Here we use idxmin functionality to handle ties in the dataset 
    #and it randomly assigns if euclideab distance results in a tie
    '''
    This function returns the index of minimum distances as a dataframe
    '''
    return pd.DataFrame(euclidean_distances.idxmin(axis=1))

        
def get_new_centroids(df_clustered_label,input_dataframe):
    '''
    The input dataframe is the dataframe with clusters labelled and the original dataframe
    '''
    df_original_label_join=input_dataframe.join(df_clustered_label)
    #This is a dataframe that consists of datapoints as well as the cluster assigned 
    df_original_label_join.rename(columns={0:'Cluster_Assigned'},inplace=True)
    #To get the new centroids we group by the Label column and take its mean
    new_centroids=df_original_label_join.groupby('Cluster_Assigned').mean()
    #Here transpose is taken to maintain consistency between original random centroids and 
    return new_centroids.T

def kmeans_llyod(input_dataframe,no_of_clusters,threshold,no_of_iterations):
    '''
    This function takes original dataframe,number of clusters,threshold as input.
    '''
    iteration=0
    #Step 1 of k means is to get random _Centroids
    initial_centroid=get_random_centroids(input_dataframe,no_of_clusters)
    #Randomly generated centroids would be stored on centroids 
    #Storing the column list to handle K ties 
    initial_centroid_column_list=initial_centroid.columns.to_list()
    
    while True:
        '''
        The while loop runs until convergence condition is met
        '''
        df_cluster_label=get_labels(input_dataframe,initial_centroid)
        df_new_centroids=get_new_centroids(df_cluster_label,input_dataframe)
        '''
        Handling (Maintaining K Centroids)
        '''
        new_list_of_columns=df_new_centroids.columns.to_list()
        #Keeping the number of clusters same
        initial_set_columns = set(initial_centroid_column_list)
        new_set_columns = set(new_list_of_columns)
        missing_columns = initial_set_columns - new_set_columns
        for col in missing_columns:
            df_new_centroids[col]=initial_centroid[col]
        
        from scipy.spatial.distance import euclidean
        scalar_product = [euclidean(initial_centroid[col],df_new_centroids[col]) for col in initial_centroid.columns]
        threshold_calculated=float(sum(scalar_product))/no_of_clusters
        
        iteration+=1
        
        if threshold_calculated<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(threshold_calculated))
        
        if iteration>no_of_iterations:
            print("Limit for iterations has exceeded")
        
        if threshold_calculated<threshold or iteration>no_of_iterations:
            return df_new_centroids
            break
        else:
            initial_centroid= df_new_centroids
        
        


# # Error with Target Variable 

# In[483]:


import numpy as np
import swifter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import time

         
def get_random_centroids(input_dataframe,no_of_clusters):
    '''
    The function takes a dataframe as an input and creates a random K centroids from uniform distribution
    '''
    #Initialize random centroids from dataset
    list_of_centroids = []
    
    for cluster in range(no_of_clusters):
        #Generates a centroids randomly from uniform distribution 
        random_centroid = input_dataframe.swifter.apply(lambda x:float(x.sample()))
        #From the given dataset it randomly selects centroids
        list_of_centroids.append(random_centroid)
    
    centroid_df=pd.concat(list_of_centroids,axis=1)
    #Naming the column as Label for ease of purpose
    centroid_df.index.name='Cluster_Assigned'
    '''
    The function returns a dataframe consisting of no of clusters required
    '''
    return centroid_df

def get_labels(input_dataframe,centroid_df):
    '''
    This function takes centroids as input and takes the initial dataframe and gives them labels to which cluster
    they belong to
    '''
    euclidean_distances = centroid_df.swifter.apply(lambda x: np.sqrt(((input_dataframe - x) ** 2).sum(axis=1)))
    #Here we use idxmin functionality to handle ties in the dataset 
    #and it randomly assigns if euclideab distance results in a tie
    '''
    This function returns the index of minimum distances as a dataframe
    '''
    return pd.DataFrame(euclidean_distances.idxmin(axis=1))

        
def get_new_centroids(df_clustered_label,input_dataframe):
    '''
    The input dataframe is the dataframe with clusters labelled and the original dataframe
    '''
    df_original_label_join=input_dataframe.join(df_clustered_label)
    #This is a dataframe that consists of datapoints as well as the cluster assigned 
    df_original_label_join.rename(columns={0:'Cluster_Assigned'},inplace=True)
    #To get the new centroids we group by the Label column and take its mean
    new_centroids=df_original_label_join.groupby('Cluster_Assigned').mean()
    #Here transpose is taken to maintain consistency between original random centroids and 
    return new_centroids.T


def kmeans_llyod(input_dataframe,no_of_clusters,threshold,no_of_iterations):
    '''
    This function takes original dataframe,number of clusters,threshold as input.
    '''
    start_time=time.time()
    iteration=0
    #Step 1 of k means is to get random _Centroids
    initial_centroid=get_random_centroids(input_dataframe,no_of_clusters)
    #Randomly generated centroids would be stored on centroids 
    #Storing the column list to handle K ties 
    initial_centroid_column_list=initial_centroid.columns.to_list()
    
    while True:
        '''
        The while loop runs until convergence condition is met
        '''
        df_cluster_label=get_labels(input_dataframe,initial_centroid)
        df_new_centroids=get_new_centroids(df_cluster_label,input_dataframe)
        '''
        Handling (Maintaining K Centroids)
        '''
        new_list_of_columns=df_new_centroids.columns.to_list()
        #Keeping the number of clusters same
        initial_set_columns = set(initial_centroid_column_list)
        new_set_columns = set(new_list_of_columns)
        missing_columns = initial_set_columns - new_set_columns
        for col in missing_columns:
            df_new_centroids[col]=initial_centroid[col]
        
        from scipy.spatial.distance import euclidean
        scalar_product = [euclidean(initial_centroid[col],df_new_centroids[col]) for col in initial_centroid.columns]
        threshold_calculated=float(sum(scalar_product))/no_of_clusters
        
        iteration+=1
        
        if threshold_calculated<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(threshold_calculated))
        
        if iteration>no_of_iterations:
            print("Limit for iterations has exceeded")
        
        if threshold_calculated<threshold or iteration>no_of_iterations:
            error=cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids)
            sum_of_square_error=sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters)
            end_time=time.time()
            return df_new_centroids,error,sum_of_square_error,end_time-start_time
            break
        else:
            initial_centroid= df_new_centroids
        

def sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters):
    '''
    This function calculates the euclidean distance between new formed 
    centroids and the datapoints in that cluster
    '''
    df_data_label=input_dataframe.join(df_cluster_label)
    #Renaming the column
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)
    total_error=[]
    for cluster in range(no_of_clusters):
        df_data_label_cluster=df_data_label[df_data_label['Cluster_Assigned']==cluster]
        df_data_label_cluster=df_data_label_cluster.drop('Cluster_Assigned',axis=1)
        centroids=pd.DataFrame(df_new_centroids[cluster])
        euclidean_distance=cdist(df_data_label_cluster,centroids.T,metric='euclidean')
        total_error.append(sum(euclidean_distance))
    return round(float(''.join(map(str, sum(total_error)))),3)
        
        
        
def cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids):
    '''
    This calculates the error for every cluster and sums up the error based on the formula for error
    '''
    
    target_variable_centroid=input_dataframe.groupby('readmitted').mean().reset_index()
    '''
    Target variable centroid is input dataframe taking mean
    '''
    new_centroids= df_new_centroids.T
    #
    df_data_label=input_dataframe.join(df_cluster_label)
    #Renaming the column
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)

    # Get the columns of the data dataframe
    columns = input_dataframe.columns

    sum_of_square_Error= []
    # Compute the distance between each data point and its assigned centroid
    for i in range(len(new_centroids)):   
        s=[]
        for j in range(len(target_variable_centroid)): ### mean centroid
            #Calculating the error between target variable centroid and new centroids
            distance = np.sum(np.square(target_variable_centroid[target_variable_centroid['readmitted']==j][columns] - new_centroids.iloc[i][columns]), axis=1)
            #Storing the distance
            s.append(distance.iloc[0])
        sum_of_square_Error.append(s)
    
    
    merged_new_label=pd.DataFrame(sum_of_square_Error).idxmin(axis=1)
    
    #Merging of cluster
    mapping_dictionary=merged_new_label.to_dict() 
    
    #Getting clusters to a new column
    df_data_label['target_variable_cluster']=df_data_label['Cluster_Assigned'].replace(mapping_dictionary)
    
    
    total_cluster_error = []
    
    for class_name in range(0,2):
        df_cluster = df_data_label[df_data_label['target_variable_cluster'] == class_name] 
        yi = len(df_cluster[df_cluster['readmitted'] == 1]) 
        #Calculating Ni
        ni = len(df_cluster[df_cluster['readmitted'] == 0]) 
        if yi == 0 and ni == 0:
            error_ci = 0
        else:
            error_ci = ni / (ni + yi) # calculate the error rate of the current cluster
        total_cluster_error.append(error_ci)
    return round(sum(total_cluster_error),3)


# # Calling K means Multiple times

# In[489]:


error_values=[]
for no_of_clusters in range(2,6):
    #Taking the cluster value from 2 to 5
    for no_of_experiments in range(1,21):
        #Performing experiments for each cluster 20 times
        final_centroids,error_target_variable,sum_of_squared_error,run_time=kmeans_llyod(df_diabetes_final,no_of_clusters,10,100)
        #Storing the variables in dataframe
        error_values.append([no_of_clusters,no_of_experiments,error_target_variable,sum_of_squared_error,run_time])
error_values_df= pd.DataFrame(error_values,columns=['No_of_Clusters', 'Iteration Number', 'Target Variable Error','Sum_of_squared_Errors','run_time'])  


# In[ ]:





# In[490]:


pd.set_option('display.max_rows',None)
error_values_df


# # Plotting the graphs of Target_Variable_Error and Sum_of_squared_errors

# In[487]:


error_plot=error_values_df.groupby(['No_of_Clusters']).mean().reset_index()[['No_of_Clusters','Target Variable Error','Sum_of_squared_Errors','run_time']]
error_plot


# In[488]:


ax = error_plot.plot(x='No_of_Clusters', y='Target Variable Error')
ax2 =error_plot.plot(x='No_of_Clusters', y='Sum_of_squared_Errors',secondary_y=True, ax=ax)
# set the axis labels and title
ax.set_xlabel('No_of_Clusters')
ax.set_ylabel('Target Variable Error')
ax2.set_ylabel('Sum_of_squared_error')
ax.set_title('Error and SSE vs No of clusters Tou')
ax.legend(['Error'], loc='upper left')
ax2.legend(['SSE'], loc='upper right')
plt.show()


# In[593]:


import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_df['No_of_Clusters'],y=error_values_df['Target Variable Error'])
plt.title('Box Plot for K means LLyod (error vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_df['No_of_Clusters'],y=error_values_df['Sum_of_squared_Errors'])
plt.title('Box Plot for K means LLyod (SSE vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_df['No_of_Clusters'],y=error_values_df['run_time'])
plt.title('Box Plot for K means LLyod (Run Time vs no of clusters)')
plt.show()


# In[595]:


import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_df['No_of_Clusters'],y=error_values_df['Sum_of_squared_Errors'])
plt.title('Box Plot for K means LLyod (SSE vs no of clusters)')
plt.show()


# In[597]:


import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_df['No_of_Clusters'],y=error_values_df['run_time'])
plt.title('Box Plot for K means LLyod (Run Time vs no of clusters)')
plt.show()


# # K means ++ Initialization Code

# In[458]:


import numpy as np
import swifter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import time

def kmeans_pp_init(input_dataframe,no_of_clusters):
    '''
    K-means++ is a variant of the K-means algorithm that aims to improve the initial centroids' selection 
    in the clustering process. 
    The standard K-means algorithm initializes the cluster centroids randomly, 
    which can lead to suboptimal clustering results, 
    especially if the dataset has complex or irregular structures.
    '''
    list_of_centroids=[]
    #Choosing the first centroid randomly
    centroid = input_dataframe.apply(lambda x: float(x.sample()))
    list_of_centroids.append(centroid)
    
    iterator=2
    while iterator<=no_of_clusters:
        '''
        Calculating the distances from the centroid to every data point
        If the no of centroids are more than 1 calculate the distance from every centroid and take minimum distance
        '''
        distances = np.array(np.amin(cdist(input_dataframe,list_of_centroids,metric='euclidean'),axis=1))
        #Next centroid will be selected with probability proportional to the distance
        
        probs = distances / np.sum(distances)
        '''
        Selection of the next centroids
        '''
        next_centroid = input_dataframe.iloc[np.random.choice(len(input_dataframe),p=probs)]
        list_of_centroids.append(next_centroid)
        iterator+=1
    
    centroid_df=pd.concat(list_of_centroids,axis=1,ignore_index=True)
    #Naming the column as Label for ease of purpose
    centroid_df.index.name='Cluster_Assigned'   
    
        
    return centroid_df

def get_labels(input_dataframe,centroid_df):
    '''
    This function takes centroids as input and takes the initial dataframe and gives them labels to which cluster
    they belong to
    '''
    euclidean_distances = centroid_df.swifter.apply(lambda x: np.sqrt(((input_dataframe - x) ** 2).sum(axis=1)))
    #Here we use idxmin functionality to handle ties in the dataset 
    #and it randomly assigns if euclideab distance results in a tie
    '''
    This function returns the index of minimum distances as a dataframe
    '''
    return pd.DataFrame(euclidean_distances.idxmin(axis=1))

        
def get_new_centroids(df_clustered_label,input_dataframe):
    '''
    The input dataframe is the dataframe with clusters labelled and the original dataframe
    '''
    df_original_label_join=input_dataframe.join(df_clustered_label)
    #This is a dataframe that consists of datapoints as well as the cluster assigned 
    df_original_label_join.rename(columns={0:'Cluster_Assigned'},inplace=True)
    #To get the new centroids we group by the Label column and take its mean
    new_centroids=df_original_label_join.groupby('Cluster_Assigned').mean()
    #Here transpose is taken to maintain consistency between original random centroids and 
    return new_centroids.T


def kmeans_plus_plus(input_dataframe,no_of_clusters,threshold,no_of_iterations):
    '''
    This function takes original dataframe,number of clusters,threshold as input.
    '''
    start_time=time.time()
    iteration=0
    #Step 1 of k means ++ is to get K means plus plus initialization centroids
    initial_centroid=kmeans_pp_init(input_dataframe,no_of_clusters)
    #Randomly generated centroids would be stored on centroids 
    #Storing the column list to handle K ties 
    initial_centroid_column_list=initial_centroid.columns.to_list()
    
    while True:
        '''
        The while loop runs until convergence condition is met
        '''
        df_cluster_label=get_labels(input_dataframe,initial_centroid)
        df_new_centroids=get_new_centroids(df_cluster_label,input_dataframe)
        '''
        Handling (Maintaining K Centroids)
        '''
        new_list_of_columns=df_new_centroids.columns.to_list()
        #Keeping the number of clusters same
        initial_set_columns = set(initial_centroid_column_list)
        new_set_columns = set(new_list_of_columns)
        missing_columns = initial_set_columns - new_set_columns
        for col in missing_columns:
            df_new_centroids[col]=initial_centroid[col]
        
        from scipy.spatial.distance import euclidean
        scalar_product = [euclidean(initial_centroid[col],df_new_centroids[col]) for col in initial_centroid.columns]
        threshold_calculated=float(sum(scalar_product))/no_of_clusters
        
        iteration+=1
        
        if threshold_calculated<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(threshold_calculated))
        
        if iteration>no_of_iterations:
            print("Limit for iterations has exceeded")
        
        if threshold_calculated<threshold or iteration>no_of_iterations:
            error=cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids)
            sum_of_square_error=sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters)
            end_time=time.time()
            return df_new_centroids,error,sum_of_square_error,end_time-start_time
            break
        else:
            initial_centroid= df_new_centroids
        

def sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters):
    '''
    This function calculates the euclidean distance between new formed 
    centroids and the datapoints in that cluster
    '''
    df_data_label=input_dataframe.join(df_cluster_label)
    #Renaming the column
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)
    total_error=[]
    for cluster in range(no_of_clusters):
        df_data_label_cluster=df_data_label[df_data_label['Cluster_Assigned']==cluster]
        df_data_label_cluster=df_data_label_cluster.drop('Cluster_Assigned',axis=1)
        centroids=pd.DataFrame(df_new_centroids[cluster])
        euclidean_distance=cdist(df_data_label_cluster,centroids.T,metric='euclidean')
        total_error.append(sum(euclidean_distance))
    return round(float(''.join(map(str, sum(total_error)))),3)
        
        
        
def cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids):
    '''
    This calculates the error for every cluster and sums up the error based on the formula for error
    '''
    
    target_variable_centroid=input_dataframe.groupby('readmitted').mean().reset_index()
    '''
    Target variable centroid is input dataframe taking mean
    '''
    new_centroids= df_new_centroids.T
    #
    df_data_label=input_dataframe.join(df_cluster_label)
    #Renaming the column
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)

    # Get the columns of the data dataframe
    columns = input_dataframe.columns

    sum_of_square_Error= []
    # Compute the distance between each data point and its assigned centroid
    for i in range(len(new_centroids)):   
        s=[]
        for j in range(len(target_variable_centroid)): ### mean centroid
            #Calculating the error between target variable centroid and new centroids
            distance = np.sum(np.square(target_variable_centroid[target_variable_centroid['readmitted']==j][columns] - new_centroids.iloc[i][columns]), axis=1)
            #Storing the distance
            s.append(distance.iloc[0])
        sum_of_square_Error.append(s)
    
    
    merged_new_label=pd.DataFrame(sum_of_square_Error).idxmin(axis=1)
    
    #Merging of cluster
    mapping_dictionary=merged_new_label.to_dict() 
    
    #Getting clusters to a new column
    df_data_label['target_variable_cluster']=df_data_label['Cluster_Assigned'].replace(mapping_dictionary)
    
    
    total_cluster_error = []
    
    for class_name in range(0,2):
        df_cluster = df_data_label[df_data_label['target_variable_cluster'] == class_name] 
        yi = len(df_cluster[df_cluster['readmitted'] == 1]) 
        #Calculating Ni
        ni = len(df_cluster[df_cluster['readmitted'] == 0]) 
        if yi == 0 and ni == 0:
            error_ci = 0
        else:
            error_ci = ni / (ni + yi) # calculate the error rate of the current cluster
        total_cluster_error.append(error_ci)
    return round(sum(total_cluster_error),3)


# # Calling K means ++ Multiple times

# In[459]:


error_values_kmeans_plus_plus=[]
for no_of_clusters in range(2,6):
    #Taking the cluster value from 2 to 5
    for no_of_experiments in range(1,21):
        #Performing experiments for each cluster 20 times
        final_centroids,error_target_variable,sum_of_squared_error,run_time=kmeans_plus_plus(df_diabetes_final,no_of_clusters,10,100)
        #Storing the variables in dataframe
        error_values_kmeans_plus_plus.append([no_of_clusters,no_of_experiments,error_target_variable,sum_of_squared_error,run_time])
error_values_kmeans_plus_plus_df= pd.DataFrame(error_values_kmeans_plus_plus,columns=['No_of_Clusters', 'Iteration Number', 'Target Variable Error','Sum_of_squared_Errors','run_time'])  


# In[461]:


error_plot_kmeans_plus_plus=error_values_kmeans_plus_plus_df.groupby(['No_of_Clusters']).mean().reset_index()[['No_of_Clusters','Target Variable Error','Sum_of_squared_Errors','run_time']]
error_plot_kmeans_plus_plus


# In[605]:


ax = error_plot_kmeans_plus_plus.plot(x='No_of_Clusters', y='Target Variable Error')
ax2=error_plot_kmeans_plus_plus.plot(x='No_of_Clusters', y='Sum_of_squared_Errors',secondary_y=True, ax=ax)
# set the axis labels and title
ax.set_xlabel('No_of_Clusters')
ax.set_ylabel('Target Variable Error')
ax2.set_ylabel('Sum_of_squared_error')
ax.set_title('Error and SSE vs No of clusters K++')
ax.legend(['Error'], loc='upper left')
ax2.legend(['SSE'], loc='upper right')
plt.show()


# In[606]:


import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_plus_plus_df['No_of_Clusters'],y=error_values_kmeans_plus_plus_df['Target Variable Error'])
plt.title('Box Plot for K means LLyod K++ (error vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_plus_plus_df['No_of_Clusters'],y=error_values_kmeans_plus_plus_df['Sum_of_squared_Errors'])
plt.title('Box Plot for K means LLyod k++(SSE vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_plus_plus_df['No_of_Clusters'],y=error_values_kmeans_plus_plus_df['run_time'])
plt.title('Box Plot for K means LLyod k++(Run Time vs no of clusters)')
plt.show()


# # Comparing error values between K means and K means ++

# In[463]:


error_plot_kmeans_plus_plus


# In[607]:


diff_error=error_plot['Sum_of_squared_Errors']-error_plot_kmeans_plus_plus['Sum_of_squared_Errors']
diff_error


# # Observations
# We can easily conclude by initializing K means using K means ++ there is a significant decrease in the Sum of Squared Errors

# # K means with SSE as convergence Criteria

# In[468]:


import numpy as np
import swifter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import time

         
def get_random_centroids(input_dataframe,no_of_clusters):
    '''
    The function takes a dataframe as an input and creates a random K centroids from uniform distribution
    '''
    #Initialize random centroids from dataset
    list_of_centroids = []
    
    for cluster in range(no_of_clusters):
        #Generates a centroids randomly from uniform distribution 
        random_centroid = input_dataframe.swifter.apply(lambda x:float(x.sample()))
        #From the given dataset it randomly selects centroids
        list_of_centroids.append(random_centroid)
    
    centroid_df=pd.concat(list_of_centroids,axis=1)
    #Naming the column as Label for ease of purpose
    centroid_df.index.name='Cluster_Assigned'
    '''
    The function returns a dataframe consisting of no of clusters required
    '''
    return centroid_df

def get_labels(input_dataframe,centroid_df):
    '''
    This function takes centroids as input and takes the initial dataframe and gives them labels to which cluster
    they belong to
    '''
    euclidean_distances = centroid_df.swifter.apply(lambda x: np.sqrt(((input_dataframe - x) ** 2).sum(axis=1)))
    #Here we use idxmin functionality to handle ties in the dataset 
    #and it randomly assigns if euclideab distance results in a tie
    '''
    This function returns the index of minimum distances as a dataframe
    '''
    return pd.DataFrame(euclidean_distances.idxmin(axis=1))

        
def get_new_centroids(df_clustered_label,input_dataframe):
    '''
    The input dataframe is the dataframe with clusters labelled and the original dataframe
    '''
    df_original_label_join=input_dataframe.join(df_clustered_label)
    #This is a dataframe that consists of datapoints as well as the cluster assigned 
    df_original_label_join.rename(columns={0:'Cluster_Assigned'},inplace=True)
    #To get the new centroids we group by the Label column and take its mean
    new_centroids=df_original_label_join.groupby('Cluster_Assigned').mean()
    #Here transpose is taken to maintain consistency between original random centroids and 
    return new_centroids.T


def kmeans_SSE_Convergence(input_dataframe,no_of_clusters,sum_of_squared_threshold,no_of_iterations):
    '''
    Treats K means as an optimization Problem and stops when difference in SSE reaches a threshold
    The input to the function is the dataframe,no of clusters and a threshold which indicates the perecentage change
    It indicates user can set the percentage change in the SSE and once the percentage change in SSE drops to the 
    Threshold we can see the algorithm has converged
    '''
    start_time=time.time()
    iteration=0
    #Step 1 of k means is to get random _Centroids
    initial_centroid=get_random_centroids(input_dataframe,no_of_clusters)
    #Randomly generated centroids would be stored on centroids 
    #Storing the column list to handle K ties 
    initial_centroid_column_list=initial_centroid.columns.to_list()
    #Get initial labels
    df_cluster_label=get_labels(input_dataframe,initial_centroid)
    #Compute the initial Sum of squared Errors
    initial_sum_of_squared_errors=sum_of_square_error_function(df_cluster_label,input_dataframe,initial_centroid,no_of_clusters)
    
    
    while True:
        '''
        The while loop runs until convergence condition is met
        '''
        
        df_new_centroids=get_new_centroids(df_cluster_label,input_dataframe)
        '''
        Handling (Maintaining K Centroids)
        '''
        new_list_of_columns=df_new_centroids.columns.to_list()
        #Keeping the number of clusters same
        initial_set_columns = set(initial_centroid_column_list)
        new_set_columns = set(new_list_of_columns)
        missing_columns = initial_set_columns - new_set_columns
        for col in missing_columns:
            df_new_centroids[col]=initial_centroid[col]
            
        '''
        Assigning labels to new centroids
        '''
        df_cluster_label_iter=get_labels(input_dataframe,df_new_centroids)
        '''
        Calculating the current SSE
        
        '''
        updated_sum_of_squared_errors=sum_of_square_error_function(df_cluster_label_iter,input_dataframe,df_new_centroids,no_of_clusters)
        
        #Calculating the convergence criteria
        
        percentage_change=((initial_sum_of_squared_errors-updated_sum_of_squared_errors)/(initial_sum_of_squared_errors))*100
        
        iteration+=1
        #Stopping criteria
        #Indicating new clusters have reduced the SSE
        if percentage_change>0:
            if percentage_change>=sum_of_squared_threshold or iteration>no_of_iterations:
                print("The input SSE Threshold was {}".format(sum_of_squared_threshold))
                print("The percentage change is {}".format(percentage_change))
                print("The initial error was {} and final error was {}".format(initial_sum_of_squared_errors,updated_sum_of_squared_errors))
                error=cluster_error_target_variable(df_cluster_label_iter,input_dataframe,no_of_clusters,df_new_centroids)
                end_time=time.time()
                return df_new_centroids,error,updated_sum_of_squared_errors,end_time-start_time
                break
                
        else:
            initial_centroid= df_new_centroids
            df_cluster_label=df_cluster_label_iter
            initial_sum_of_squared_errors=updated_sum_of_squared_errors
        

def sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters):
    '''
    This function calculates the euclidean distance between new formed 
    centroids and the datapoints in that cluster
    '''
    df_data_label=input_dataframe.join(df_cluster_label)
    #Renaming the column
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)
    total_error=[]
    for cluster in range(no_of_clusters):
        df_data_label_cluster=df_data_label[df_data_label['Cluster_Assigned']==cluster]
        df_data_label_cluster=df_data_label_cluster.drop('Cluster_Assigned',axis=1)
        centroids=pd.DataFrame(df_new_centroids[cluster])
        euclidean_distance=cdist(df_data_label_cluster,centroids.T,metric='euclidean')
        total_error.append(sum(euclidean_distance))
    return round(float(''.join(map(str, sum(total_error)))),3)
        
        
        
def cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids):
    '''
    This calculates the error for every cluster and sums up the error based on the formula for error
    '''
    
    target_variable_centroid=input_dataframe.groupby('readmitted').mean().reset_index()
    '''
    Target variable centroid is input dataframe taking mean
    '''
    new_centroids= df_new_centroids.T
    #
    df_data_label=input_dataframe.join(df_cluster_label)
    #Renaming the column
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)

    # Get the columns of the data dataframe
    columns = input_dataframe.columns

    sum_of_square_Error= []
    # Compute the distance between each data point and its assigned centroid
    for i in range(len(new_centroids)):   
        s=[]
        for j in range(len(target_variable_centroid)): ### mean centroid
            #Calculating the error between target variable centroid and new centroids
            distance = np.sum(np.square(target_variable_centroid[target_variable_centroid['readmitted']==j][columns] - new_centroids.iloc[i][columns]), axis=1)
            #Storing the distance
            s.append(distance.iloc[0])
        sum_of_square_Error.append(s)
    
    
    merged_new_label=pd.DataFrame(sum_of_square_Error).idxmin(axis=1)
    
    #Merging of cluster
    mapping_dictionary=merged_new_label.to_dict() 
    
    #Getting clusters to a new column
    df_data_label['target_variable_cluster']=df_data_label['Cluster_Assigned'].replace(mapping_dictionary)
    
    
    total_cluster_error = []
    
    for class_name in range(0,2):
        df_cluster = df_data_label[df_data_label['target_variable_cluster'] == class_name] 
        yi = len(df_cluster[df_cluster['readmitted'] == 1]) 
        #Calculating Ni
        ni = len(df_cluster[df_cluster['readmitted'] == 0]) 
        if yi == 0 and ni == 0:
            error_ci = 0
        else:
            error_ci = ni / (ni + yi) # calculate the error rate of the current cluster
        total_cluster_error.append(error_ci)
    return round(sum(total_cluster_error),3)


# # Calling K means with convergence multiple times

# In[469]:


error_values_kmeans_convergence=[]
for no_of_clusters in range(2,6):
    #Taking the cluster value from 2 to 5
    for no_of_experiments in range(1,21):
        #Performing experiments for each cluster 20 times
        final_centroids,error_target_variable,sum_of_squared_error,run_time=kmeans_SSE_Convergence(df_diabetes_final,no_of_clusters,10,100)
        #Storing the variables in dataframe
        error_values_kmeans_convergence.append([no_of_clusters,no_of_experiments,error_target_variable,sum_of_squared_error,run_time])
error_values_kmeans_convergence_df= pd.DataFrame(error_values_kmeans_convergence,columns=['No_of_Clusters', 'Iteration Number', 'Target Variable Error','Sum_of_squared_Errors','run_time'])  


# In[598]:


error_values_kmeans_convergence_df


# In[599]:


error_values_kmeans_convergence=error_values_kmeans_convergence_df.groupby(['No_of_Clusters']).mean().reset_index()[['No_of_Clusters','Target Variable Error','Sum_of_squared_Errors','run_time']]
error_values_kmeans_convergence


# In[602]:


ax = error_values_kmeans_convergence.plot(x='No_of_Clusters', y='Sum_of_squared_Errors')
ax2=error_values_kmeans_convergence.plot(x='No_of_Clusters', y='Target Variable Error',secondary_y=True, ax=ax)
# set the axis labels and title
ax.set_xlabel('No_of_Clusters')
ax.set_ylabel('Target Variable Error')
ax2.set_ylabel('Sum_of_squared_error')
ax.set_title('Error and SSE vs No of clusters SSE Convergence')
ax.legend(['SSE'], loc='upper left')
ax2.legend(['Error'], loc='upper right')
plt.show()


# In[604]:


import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_convergence_df['No_of_Clusters'],y=error_values_kmeans_convergence_df['Target Variable Error'])
plt.title('Box Plot for K means LLyod SSE (error vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_convergence_df['No_of_Clusters'],y=error_values_kmeans_convergence_df['Sum_of_squared_Errors'])
plt.title('Box Plot for K means LLyod SSE(SSE vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_convergence_df['No_of_Clusters'],y=error_values_kmeans_convergence_df['run_time'])
plt.title('Box Plot for K means LLyod SSE (Run Time vs no of clusters)')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Run time comparison between K means and K means++

# In[472]:


error_values_kmeans_convergence['Algorithm_Used']='K_Means_SSE_Convergence'
error_plot_kmeans_plus_plus['Algorithm_Used']='K++Initialization'
error_plot['Algorithm_Used']='Llyod_Kmeans'


# In[473]:


k_means_metric=pd.concat([error_values_kmeans_convergence,error_plot_kmeans_plus_plus,error_plot],axis=0)


# In[474]:


k_means_metric


# In[612]:


sns.lineplot(x="No_of_Clusters", y="Sum_of_squared_Errors", hue="Algorithm_Used", data=k_means_metric)
plt.title('Diabetes_Dataset_Comparison')
plt.show()
sns.barplot(x="No_of_Clusters", y="Sum_of_squared_Errors", hue="Algorithm_Used", data=k_means_metric)
plt.title('Diabetes_Dataset_SSE_Comparison')
plt.show()
sns.barplot(x="No_of_Clusters", y="Target Variable Error", hue="Algorithm_Used", data=k_means_metric)
plt.title('Diabetes_dataset_eror_comparison')
plt.show()
sns.lineplot(x="No_of_Clusters", y="run_time", hue="Algorithm_Used", data=k_means_metric)
plt.title('Diabetes_Dataset_Comparison')
plt.show()


# # K means with library

# In[482]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

# Load data from CSV file
data = df_diabetes_final
columns=df_diabetes_final.columns

#Extract features
X = data[['age', 'admission_type_id', 'discharge_disposition_id','admission_source_id', 'time_in_hospital', 'num_lab_procedures','num_procedures', 'num_medications','number_outpatient','number_emergency', 
        'number_inpatient', 'number_diagnoses','max_glu_serum', 'A1Cresult', 'metformin', 'glimepiride', 'glipizide',
       'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'change',
       'diabetesMed', 'readmitted', 'gender_Female', 'gender_Male',
       'race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 'race_Hispanic',
       'race_Other']]

# Create a list to hold the Sum of Squared Distances (SSD)
ssd = []

# Create KMeans objects for k=1 to k=10
for k in tqdm(range(1, 16)):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    ssd.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(1,16), ssd)
plt.title('Elbow Curve of Diabetes Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('SSD')
plt.show()


# # Ball K means

# In[492]:


df_sample=df_diabetes_final[['readmitted','num_medications','num_lab_procedures']]
df_sample=df_sample.head(10)


# In[504]:


import numpy as np
import swifter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import time

         
def get_random_centroids(input_dataframe,no_of_clusters):
    '''
    The function takes a dataframe as an input and creates a random K centroids from uniform distribution
    '''
    #Initialize random centroids from dataset
    list_of_centroids = []
    
    for cluster in range(no_of_clusters):
        #Generates a centroids randomly from uniform distribution 
        random_centroid = input_dataframe.swifter.apply(lambda x:float(x.sample()))
        #From the given dataset it randomly selects centroids
        list_of_centroids.append(random_centroid)
    
    centroid_df=pd.concat(list_of_centroids,axis=1)
    #Naming the column as Label for ease of purpose
    centroid_df.index.name='Cluster_Assigned'
    '''
    The function returns a dataframe consisting of no of clusters required
    '''
    return centroid_df

def get_labels(input_dataframe,centroid_df):
    '''
    This function takes centroids as input and takes the initial dataframe and gives them labels to which cluster
    they belong to
    '''
    euclidean_distances = centroid_df.swifter.apply(lambda x: np.sqrt(((input_dataframe - x) ** 2).sum(axis=1)))
    #Here we use idxmin functionality to handle ties in the dataset 
    #and it randomly assigns if euclideab distance results in a tie
    '''
    This function returns the index of minimum distances as a dataframe
    '''
    return pd.DataFrame(euclidean_distances.idxmin(axis=1))

        
def get_new_centroids(df_clustered_label,input_dataframe):
    '''
    The input dataframe is the dataframe with clusters labelled and the original dataframe
    '''
    df_original_label_join=input_dataframe.join(df_clustered_label)
    #This is a dataframe that consists of datapoints as well as the cluster assigned 
    df_original_label_join.rename(columns={0:'Cluster_Assigned'},inplace=True)
    #To get the new centroids we group by the Label column and take its mean
    new_centroids=df_original_label_join.groupby('Cluster_Assigned').mean()
    #Here transpose is taken to maintain consistency between original random centroids and 
    return new_centroids.T


def kmeans_llyod(input_dataframe,no_of_clusters,threshold,no_of_iterations):
    '''
    This function takes original dataframe,number of clusters,threshold as input.
    '''
    start_time=time.time()
    iteration=0
    #Step 1 of k means is to get random _Centroids
    initial_centroid=get_random_centroids(input_dataframe,no_of_clusters)
    #Randomly generated centroids would be stored on centroids 
    #Storing the column list to handle K ties 
    initial_centroid_column_list=initial_centroid.columns.to_list()
    
    while True:
        '''
        The while loop runs until convergence condition is met
        '''
        df_cluster_label=get_labels(input_dataframe,initial_centroid)
        df_new_centroids=get_new_centroids(df_cluster_label,input_dataframe)
        '''
        Handling (Maintaining K Centroids)
        '''
        new_list_of_columns=df_new_centroids.columns.to_list()
        #Keeping the number of clusters same
        initial_set_columns = set(initial_centroid_column_list)
        new_set_columns = set(new_list_of_columns)
        missing_columns = initial_set_columns - new_set_columns
        for col in missing_columns:
            df_new_centroids[col]=initial_centroid[col]
        
        from scipy.spatial.distance import euclidean
        scalar_product = [euclidean(initial_centroid[col],df_new_centroids[col]) for col in initial_centroid.columns]
        threshold_calculated=float(sum(scalar_product))/no_of_clusters
        
        iteration+=1
        
        if threshold_calculated<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(threshold_calculated))
        
        if iteration>no_of_iterations:
            print("Limit for iterations has exceeded")
        
        if threshold_calculated<threshold or iteration>no_of_iterations:
            error=cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids)
            sum_of_square_error=sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters)
            end_time=time.time()
            return df_new_centroids,error,sum_of_square_error,end_time-start_time
            break
        else:
            initial_centroid= df_new_centroids
        

def sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters):
    '''
    This function calculates the euclidean distance between new formed 
    centroids and the datapoints in that cluster
    '''
    df_data_label=input_dataframe.join(df_cluster_label)
    #Renaming the column
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)
    total_error=[]
    for cluster in range(no_of_clusters):
        df_data_label_cluster=df_data_label[df_data_label['Cluster_Assigned']==cluster]
        df_data_label_cluster=df_data_label_cluster.drop('Cluster_Assigned',axis=1)
        centroids=pd.DataFrame(df_new_centroids[cluster])
        euclidean_distance=cdist(df_data_label_cluster,centroids.T,metric='euclidean')
        total_error.append(sum(euclidean_distance))
    return round(float(''.join(map(str, sum(total_error)))),3)
        
        
        
def cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids):
    '''
    This calculates the error for every cluster and sums up the error based on the formula for error
    '''
    
    target_variable_centroid=input_dataframe.groupby('readmitted').mean().reset_index()
    '''
    Target variable centroid is input dataframe taking mean
    '''
    new_centroids= df_new_centroids.T
    #
    df_data_label=input_dataframe.join(df_cluster_label)
    #Renaming the column
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)

    # Get the columns of the data dataframe
    columns = input_dataframe.columns

    sum_of_square_Error= []
    # Compute the distance between each data point and its assigned centroid
    for i in range(len(new_centroids)):   
        s=[]
        for j in range(len(target_variable_centroid)): ### mean centroid
            #Calculating the error between target variable centroid and new centroids
            distance = np.sum(np.square(target_variable_centroid[target_variable_centroid['readmitted']==j][columns] - new_centroids.iloc[i][columns]), axis=1)
            #Storing the distance
            s.append(distance.iloc[0])
        sum_of_square_Error.append(s)
    
    
    merged_new_label=pd.DataFrame(sum_of_square_Error).idxmin(axis=1)
    
    #Merging of cluster
    mapping_dictionary=merged_new_label.to_dict() 
    
    #Getting clusters to a new column
    df_data_label['target_variable_cluster']=df_data_label['Cluster_Assigned'].replace(mapping_dictionary)
    
    
    total_cluster_error = []
    
    for class_name in range(0,2):
        df_cluster = df_data_label[df_data_label['target_variable_cluster'] == class_name] 
        yi = len(df_cluster[df_cluster['readmitted'] == 1]) 
        #Calculating Ni
        ni = len(df_cluster[df_cluster['readmitted'] == 0]) 
        if yi == 0 and ni == 0:
            error_ci = 0
        else:
            error_ci = ni / (ni + yi) # calculate the error rate of the current cluster
        total_cluster_error.append(error_ci)
    return round(sum(total_cluster_error),3)


# In[538]:


df_new_centroids,error,sum_of_square_error,x,y,z=kmeans_llyod(df_sample,3,10,100)


# In[539]:


df_new_centroids


# In[579]:


dist=[]
for i in range(0,3):
    distance=cdist(pd.DataFrame(df_new_centroids[i]).T,df_sample,metric='Euclidean')
    dist.append(distance)
dict={}    
for index,i in enumerate(dist):
    for j in i:
        dict[index]=max(j)  
centroid_df_radius=pd.DataFrame(dict,index=list(dict.keys()))
centroid_df_radius.drop_duplicates(inplace=True)
centroid_df_radius


# In[581]:


centroid_df_neighbour=df_new_centroids.copy()


# In[584]:


centroid_df_neighbour


# In[589]:


for i in range(0,3):
    for j in range(i+1,3):
        print(i,j)
        print(i,j,cdist(df_new_centroids[i],df_new_centroids[j].T,method='euclidean'))


# In[ ]:




