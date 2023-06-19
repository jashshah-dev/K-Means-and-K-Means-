#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import swifter
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('wordnet')
nltk.download('stopwords')


# In[3]:


import swifter


# In[4]:


df_new_york=pd.read_csv('nyt-comments-2020.csv',low_memory=False)


# In[5]:


import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer

def preprocess(corpus):
    #Converting text to lower case
    word_tokens_initialize=word_tokenize(corpus)
    corpus = corpus.lower()
    # Remove punctuation
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    #remove numbers
    corpus = re.sub(r'\d+', '', corpus)
    #Reomve URLS
    corpus = re.sub(r'http\S+', '',corpus)
    # Remove non-alphabetic
    corpus = re.sub(r'[^a-zA-Z0-9\s]', '',corpus)
    #Remove Stop words
    word_tokens=word_tokenize(corpus)
    stop_words = set(stopwords.words('english'))
    word_tokens = [word for word in word_tokens if word not in stop_words]
    #Applying Stemming
    stemmer = PorterStemmer()
    word_tokens=[stemmer.stem(word) for word in word_tokens]
    corpus = ' '.join(word_tokens)
    return corpus
    


# In[6]:


df_new_york['Cleaned_data']=df_new_york['commentBody'].swifter.apply(lambda x: preprocess(x))


# In[10]:


df_new_york.to_csv('Final_data.csv')


# In[113]:


df_first_copy=df_new_york.copy()


# In[78]:


df_new_york.head(10)


# In[ ]:


df_new_york['commentType'].value_counts()


# In[82]:


df_first_copy=df_new_york[df_new_york['commentType']=='comment']


# In[114]:


df_new_york_final=df_first_copy[['Cleaned_data','editorsSelection','commentBody']]


# In[115]:


df_new_york_final_01=df_new_york_final.copy()
df_new_york_final_01.shape


# In[117]:


import numpy as np
df_new_york_final_01['Cleaned_data'].replace({'':np.nan},inplace=True)


# In[118]:


df_new_york_final_01=df_new_york_final_01.dropna()


# # Applying Count Vectorizer to convert in matrix form Threshold 2.95 Percent

# In[120]:


from sklearn.feature_extraction.text import CountVectorizer
num_docs = len(df_new_york_final_01)
min_df_pct =0.0295
min_df = int(min_df_pct * num_docs)
min_df


# In[121]:


from tqdm import tqdm
#To chcek the percentage bar
from sklearn.feature_extraction.text import CountVectorizer
#Applying count vectorizer with min df as 2.7 percent
vectorizer = CountVectorizer(min_df=min_df)
#Hence we have only kept those key words whos minimum occurence is 2.7 percent in data
bag_of_words_matrix = vectorizer.fit_transform(tqdm(df_new_york_final_01['Cleaned_data']))
count_vectorizer_df= pd.DataFrame.sparse.from_spmatrix(bag_of_words_matrix, columns=vectorizer.get_feature_names())


# In[208]:


count_vectorizer_df.shape


# In[123]:


count_vectorizer_df['editorsSelection']=df_new_york_final_01['editorsSelection']


# In[102]:


count_vectorizer_df['editorsSelection'].value_counts()


# In[124]:


count_vectorizer_df['editorsSelection'] = count_vectorizer_df['editorsSelection'].replace({True:1,False:0})


# In[222]:


count_vectorizer_df.shape


# In[196]:


df_sample=count_vectorizer_df.copy()
df_sample=df_sample.head(1000000)


# # K means using TOU and Target Variable Error

# In[213]:


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
        total_error.append(np.nansum(euclidean_distance))
    
    return round(np.nansum(total_error),3)
    #return round(float(''.join(map(str, sum(total_error)))),3)
        
        
        
def cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids):
    '''
    This calculates the error for every cluster and sums up the error based on the formula for error
    '''
    
    target_variable_centroid=input_dataframe.groupby('editorsSelection').mean().reset_index()
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
            distance = np.sum(np.square(target_variable_centroid[target_variable_centroid['editorsSelection']==j][columns] - new_centroids.iloc[i][columns]), axis=1)
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
        yi = len(df_cluster[df_cluster['editorsSelection'] == 1]) 
        #Calculating Ni
        ni = len(df_cluster[df_cluster['editorsSelection'] == 0]) 
        if yi == 0 and ni == 0:
            error_ci = 0
        else:
            error_ci = ni / (ni + yi) # calculate the error rate of the current cluster
        total_cluster_error.append(error_ci)
    return round(sum(total_cluster_error),3)


# # Running K means Llyod Multiple times

# In[ ]:





# In[254]:


df_sample=count_vectorizer_df.copy()
df_sample=df_sample.head(100000)


# In[255]:


error_values=[]
for no_of_clusters in range(2,6):
    #Taking the cluster value from 2 to 5
    for no_of_experiments in range(1,21):
        #Performing experiments for each cluster 20 times
        final_centroids,error_target_variable,sum_of_squared_error,run_time=kmeans_llyod(df_sample,no_of_clusters,10,100)
        #Storing the variables in dataframe
        error_values.append([no_of_clusters,no_of_experiments,error_target_variable,sum_of_squared_error,run_time])
error_values_df= pd.DataFrame(error_values,columns=['No_of_Clusters', 'Iteration Number', 'Target Variable Error','Sum_of_squared_Errors','run_time'])  
error_values_df.to_csv('Kmeans_llyod_20_iteration_1lakh.csv')


# # Done both the dataframes

# In[264]:


error_values_df


# In[268]:


error_plot=error_values_df.groupby(['No_of_Clusters']).mean().reset_index()[['No_of_Clusters','Target Variable Error','Sum_of_squared_Errors','run_time']]
error_plot


# In[305]:


ax = error_plot.plot(x='No_of_Clusters', y='Target Variable Error')
ax2 =error_plot.plot(x='No_of_Clusters', y='Sum_of_squared_Errors',secondary_y=True, ax=ax)
# set the axis labels and title
ax.set_xlabel('No_of_Clusters')
ax.set_ylabel('Target Variable Error')
ax2.set_ylabel('Sum_of_squared_error')
ax.set_title('Error and SSE vs No of clusters Tou In NYC')
ax.legend(['Error'], loc='upper left')
ax2.legend(['SSE'], loc='upper right')

plt.show()


# In[302]:


import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_df['No_of_Clusters'],y=error_values_df['Target Variable Error'])
plt.title('Box Plot for K means LLyod nyc(error vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_df['No_of_Clusters'],y=error_values_df['Sum_of_squared_Errors'])
plt.title('Box Plot for K means LLyod nyc(SSE vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_df['No_of_Clusters'],y=error_values_df['run_time'])
plt.title('Box Plot for K means LLyod NYC (Run Time vs no of clusters)')
plt.show()


# # K means Plus Plus 

# In[232]:


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
        
        probs = distances / np.nansum(distances)
        probs = [0 if np.isnan(x) else x for x in probs]
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


def kmeans_llyod_kpp(input_dataframe,no_of_clusters,threshold,no_of_iterations):
    '''
    This function takes original dataframe,number of clusters,threshold as input.
    '''
    start_time=time.time()
    iteration=0
    #Step 1 of k means is to get random _Centroids
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
        total_error.append(np.nansum(euclidean_distance))
    
    return round(np.nansum(total_error),3)
    #return round(float(''.join(map(str, sum(total_error)))),3)
        
        
        
def cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids):
    '''
    This calculates the error for every cluster and sums up the error based on the formula for error
    '''
    
    target_variable_centroid=input_dataframe.groupby('editorsSelection').mean().reset_index()
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
            distance = np.sum(np.square(target_variable_centroid[target_variable_centroid['editorsSelection']==j][columns] - new_centroids.iloc[i][columns]), axis=1)
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
        yi = len(df_cluster[df_cluster['editorsSelection'] == 1]) 
        #Calculating Ni
        ni = len(df_cluster[df_cluster['editorsSelection'] == 0]) 
        if yi == 0 and ni == 0:
            error_ci = 0
        else:
            error_ci = ni / (ni + yi) # calculate the error rate of the current cluster
        total_cluster_error.append(error_ci)
    return round(sum(total_cluster_error),3)


# In[ ]:





# In[309]:


error_values_kmeans_plus_plus=[]
for no_of_clusters in range(2,6):
    #Taking the cluster value from 2 to 5
    for no_of_experiments in range(1,21):
        #Performing experiments for each cluster 20 times
        final_centroids,error_target_variable,sum_of_squared_error,run_time=kmeans_llyod_kpp(count_vectorizer_df,no_of_clusters,10,100)
        #Storing the variables in dataframe
        error_values_kmeans_plus_plus.append([no_of_clusters,no_of_experiments,error_target_variable,sum_of_squared_error,run_time])
error_values_kmeans_plus_plus_df= pd.DataFrame(error_values_kmeans_plus_plus,columns=['No_of_Clusters', 'Iteration Number', 'Target Variable Error','Sum_of_squared_Errors','run_time'])  
error_values_kmeans_plus_plus_df.to_csv('KPP_NYC_Full_data_02.csv')


# # Done both the dataframes

# In[312]:


error_values_kmeans_plus_plus_df


# In[313]:


error_plot_kmeans_plus_plus=error_values_kmeans_plus_plus_df.groupby(['No_of_Clusters']).mean().reset_index()[['No_of_Clusters','Target Variable Error','Sum_of_squared_Errors','run_time']]
error_plot_kmeans_plus_plus


# In[314]:


ax = error_plot_kmeans_plus_plus.plot(x='No_of_Clusters', y='Target Variable Error')
ax2=error_plot_kmeans_plus_plus.plot(x='No_of_Clusters', y='Sum_of_squared_Errors',secondary_y=True, ax=ax)
# set the axis labels and title
ax.set_xlabel('No_of_Clusters')
ax.set_ylabel('Target Variable Error')
ax2.set_ylabel('Sum_of_squared_error')
ax.set_title('Error and SSE vs No of clusters nyc K++')
ax.legend(['Error'], loc='upper left')
ax2.legend(['SSE'], loc='upper right')

plt.show()


# In[315]:


import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_plus_plus_df['No_of_Clusters'],y=error_values_kmeans_plus_plus_df['Target Variable Error'])
plt.title('Box Plot for K means LLyod K++ NYC(error vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_plus_plus_df['No_of_Clusters'],y=error_values_kmeans_plus_plus_df['Sum_of_squared_Errors'])
plt.title('Box Plot for K means LLyod k++( NYC SSE vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_plus_plus_df['No_of_Clusters'],y=error_values_kmeans_plus_plus_df['run_time'])
plt.title('Box Plot for K means LLyod k++ NYC(Run Time vs no of clusters)')
plt.show()
sns.lineplot(x="No_of_Clusters", y="Sum_of_squared_Errors", hue="Algorithm_Used", data=k_means_metric)
plt.title('NYC_Comparison')
plt.show()
sns.barplot(x="No_of_Clusters", y="Sum_of_squared_Errors", hue="Algorithm_Used", data=k_means_metric)
plt.title('NYC_Comparison')
plt.show()
sns.barplot(x="No_of_Clusters", y="Target Variable Error", hue="Algorithm_Used", data=k_means_metric)
plt.title('NYC_Comparison')
plt.show()
sns.lineplot(x="No_of_Clusters", y="run_time", hue="Algorithm_Used", data=k_means_metric)
plt.title('NYC_Comparison')
plt.show()


# # Entire Dataset 

# In[279]:


df=pd.read_csv('KPP_NYC.csv')


# In[282]:


df.plot(x='No_of_Clusters', y='Sum_of_squared_Errors')
plt.title('Entire Dataset of NYC SSE K++')
plt.show()


# In[270]:


error_values_kmeans_convergence['Algorithm_Used']='K_Means_SSE_Convergence'
error_plot_kmeans_plus_plus['Algorithm_Used']='K++Initialization'
error_plot['Algorithm_Used']='Llyod_Kmeans'


# In[271]:


k_means_metric=pd.concat([error_values_kmeans_convergence,error_plot_kmeans_plus_plus,error_plot],axis=0)


# In[272]:


k_means_metric


# In[ ]:





# In[ ]:





# # K means using SSE

# In[260]:


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
        total_error.append(np.nansum(euclidean_distance))
    
    return round(np.nansum(total_error),3)
        
        
        
def cluster_error_target_variable(df_cluster_label,input_dataframe,no_of_clusters,df_new_centroids):
    '''
    This calculates the error for every cluster and sums up the error based on the formula for error
    '''
    
    target_variable_centroid=input_dataframe.groupby('editorsSelection').mean().reset_index()
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
            distance = np.sum(np.square(target_variable_centroid[target_variable_centroid['editorsSelection']==j][columns] - new_centroids.iloc[i][columns]), axis=1)
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
        yi = len(df_cluster[df_cluster['editorsSelection'] == 1]) 
        #Calculating Ni
        ni = len(df_cluster[df_cluster['editorsSelection'] == 0]) 
        if yi == 0 and ni == 0:
            error_ci = 0
        else:
            error_ci = ni / (ni + yi) # calculate the error rate of the current cluster
        total_cluster_error.append(error_ci)
    return round(sum(total_cluster_error),3)


# In[311]:


error_values_kmeans_convergence=[]
for no_of_clusters in range(2,6):
    #Taking the cluster value from 2 to 5
    for no_of_experiments in range(1,21):
        #Performing experiments for each cluster 20 times
        final_centroids,error_target_variable,sum_of_squared_error,run_time=kmeans_SSE_Convergence(count_vectorizer_df,no_of_clusters,10,100)
        #Storing the variables in dataframe
        error_values_kmeans_convergence.append([no_of_clusters,no_of_experiments,error_target_variable,sum_of_squared_error,run_time])
error_values_kmeans_convergence_df_full= pd.DataFrame(error_values_kmeans_convergence,columns=['No_of_Clusters', 'Iteration Number', 'Target Variable Error','Sum_of_squared_Errors','run_time'])  
error_values_kmeans_convergence_df_full.to_csv('nyc_Sse_data_fukk_data_03_march.csv')


# # Done Both Dataframes

# In[262]:


error_values_kmeans_convergence_df


# In[269]:


error_values_kmeans_convergence=error_values_kmeans_convergence_df.groupby(['No_of_Clusters']).mean().reset_index()[['No_of_Clusters','Target Variable Error','Sum_of_squared_Errors','run_time']]
error_values_kmeans_convergence


# # Plots

# In[307]:


ax = error_values_kmeans_convergence.plot(x='No_of_Clusters', y='Target Variable Error')
ax2 =error_values_kmeans_convergence.plot(x='No_of_Clusters', y='Sum_of_squared_Errors',secondary_y=True, ax=ax)
# set the axis labels and title
ax.set_xlabel('No_of_Clusters')
ax.set_ylabel('Target Variable Error')
ax2.set_ylabel('Sum_of_squared_error')
ax.set_title('Error and SSE vs No of clusters SSE In NYC')
ax.legend(['Error'], loc='upper left')
ax2.legend(['SSE'], loc='upper right')

plt.show()


# In[284]:


import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_convergence_df['No_of_Clusters'],y=error_values_kmeans_convergence_df['Target Variable Error'])
plt.title('Box Plot for K means LLyod SSE NYC(error vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_convergence_df['No_of_Clusters'],y=error_values_kmeans_convergence_df['Sum_of_squared_Errors'])
plt.title('Box Plot for K means LLyod SSE NYC(SSE vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 10))
#Plotting Box plot
#Plotting values of errors for 80 iterations
sns.boxplot(x=error_values_kmeans_convergence_df['No_of_Clusters'],y=error_values_kmeans_convergence_df['run_time'])
plt.title('Box Plot for K means LLyod SSE NYC (Run Time vs no of clusters)')
plt.show()


# # Entire Dataset

# In[297]:


df=pd.read_csv('nyc_Sse_data.csv')
df.plot(x='No_of_Clusters',y='Sum_of_squared_Errors')
plt.title('NYC_DATASET_SSE')


# In[238]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

# Load data from CSV file
data = count_vectorizer_df
data=data.fillna(0)

# Create a list to hold the Sum of Squared Distances (SSD)
ssd = []

# Create KMeans objects for k=2 to k=16
for k in tqdm(range(2, 16)):
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(data)
    ssd.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(2, 16), ssd)
plt.title('Elbow Curve of New York Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('SSD')
plt.show()




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[69]:





# In[ ]:





# In[32]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




