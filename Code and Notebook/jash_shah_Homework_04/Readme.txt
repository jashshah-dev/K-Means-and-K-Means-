The dependencies for running ipynb/python files are:
Following libraries must be downloaded
numpy 
For mathematic Calculations
math
itertools
For plotting Graphs
matplotlib
seaborn
For Natural Language Processing
string
nltk-Natural Language processing
nltk.corpus
stopwords
nltk.stem
PorterStemmer
nltk.tokenize
word_tokenize
sklearn.feature_extraction.text
CountVectorizer
tqdm-To check the status of running
swifter-Is a library to parallelize run time
Download the words for stemming and stop words eliminatiom
nltk.download('wordnet')

nltk.download('stopwords')
#Downloading punctuations
nltk.download('punkt')


Execution Guidelines:
We have used Big Red 200 Servers to run the code and for Comments Dataset it takes roughly 20 hours to complete 80 iterations.

Dataset Details:
For diabetes dataset I have implemented EDA on it before applying clustering algorithms
For NYC Comments dataset I have selected top 2.9 percent of words to focus on most occuring words.
After performing clustering we can check which words are used the most and which words form a cluster.
After applying NLP Preprocessing steps have created a Count Vectorizer Dataframe and transformed the text data into appropriate vector representation.

For running the dataset.
1)Run the code and after running we can check the clusters 
2)The output dataframe consists of No of Iterations, Cluster Number, Target_Variable_Error,
Run_time,Sum_of_squared_error
3) To plot the graph we groupby the cluster number and take the mean 
4) After taking the mean value we are able to plot the graphs







