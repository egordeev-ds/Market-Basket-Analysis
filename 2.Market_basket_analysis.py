# MARKET BASKET ANALYSES
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, ArrayType, DoubleType
from pyspark.sql.functions import countDistinct, collect_set, udf, col, count, collect_set, lit, when, concat_ws, substring, length, col, expr, split, size

import seaborn as sns
import matplotlib.pyplot as plt

from datetime import date, datetime

import pandas as pd
import numpy as np
from wordcloud import WordCloud

import itertools
from operator import add

import warnings
warnings.filterwarnings('ignore')

# 1.Introduction

### Download datates via kaggle API
# !pip install kaggle
# !pip install -q kaggle

# !mkdir -p ./Kaggle
# !cp kaggle.json ./Kaggle #need to put personal kaggle.json file from kaggle website to working directory
# os.environ['KAGGLE_CONFIG_DIR'] = "./Kaggle"

# !kaggle datasets download -d ashirwadsangwan/imdb-dataset
# !unzip imdb-dataset.zip

## Initialize PySpark session

# initialize spark session
conf = SparkConf().setAppName("IMDB_MarketBasketAnalysis")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '32G')
        .set('spark.driver.memory', '128G')
        .set('spark.driver.maxResultSize', '24G'))

sc = SparkContext(conf= conf)
spark = SparkSession.builder.appName("IMDB_MarketBasketAnalysis").getOrCreate()

# download datasets to local memmory
principals_path = '/Users/cotangentofzero/DATA_SCIENCE/Students/Natali/7.Data/title.principals.tsv.gz'
movies_path = '/Users/cotangentofzero/DATA_SCIENCE/Students/Natali/7.Data/title.basics.tsv.gz'
persons_path = '/Users/cotangentofzero/DATA_SCIENCE/Students/Natali/7.Data/name.basics.tsv.gz'

principals = spark.read.csv(principals_path, sep='\t', header=True, inferSchema=True)
movies = spark.read.csv(movies_path, sep= '\t', header=True, inferSchema=True)
persons = spark.read.csv(persons_path, sep='\t' , header=True, inferSchema=True)

#create a temp view
persons.createOrReplaceTempView('persons')
principals.createOrReplaceTempView('principals')
movies.createOrReplaceTempView('movies')

## Datasets

# view each table separately
print('principals')
principals.show(5)
print('movies')
movies.show(5)
print('persons')
persons.show(5)

## Data exploration

# checking if there are missings in important columns
print('movies_titleType',movies.filter(col('titleType').isNull()).count())
print('movies_primaryTitle:',movies.filter(col('primaryTitle').isNull()).count())

print('persons_primaryName:',persons.filter(col('primaryName').isNull()).count())
print('principals_category:',principals.filter(col('category').isNull()).count())

# checking the data types
print('principals')
principals.printSchema()
print('movies')
movies.printSchema()
print('persons')
persons.printSchema()

# construct a table: actor and all the movies he's played to
query_movies_and_actors = """SELECT movies.primaryTitle, persons.primaryName
                             FROM movies
                             INNER JOIN principals
                             ON principals.tconst = movies.tconst
                             INNER JOIN persons
                             ON persons.nconst = principals.nconst
                             WHERE (category = 'actor' or category = 'actress') and (movies.titleType = 'movie');
                             """
movies_and_actors = spark.sql(query_movies_and_actors)
movies_and_actors.createOrReplaceTempView('movies_and_actors')
movies_and_actors.show(5)

# 2.Finiding insights

## Statistics
print('rows in dataset:', movies_and_actors.count())
print('unique actors in the dataset:', movies_and_actors.select('primaryName').distinct().count())
print('unique movies in the dataset:', movies_and_actors.select('primaryTitle').distinct().count())

## Top 10 actors with the biggest number of movies

#construct a table to see actors with the biggest number of movies
query_actors_with_top_n_of_movies = """SELECT primaryName, COUNT(*) AS n_movies
                                       FROM movies_and_actors
                                       GROUP BY primaryName
                                       ORDER BY n_movies desc
                                       """

actors_with_top_n_of_movies = spark.sql(query_actors_with_top_n_of_movies)
actors_with_top_n_of_movies.createOrReplaceTempView('actors_with_top_n_of_movies')
print('actors_with_top_n_of_movies')
actors_with_top_n_of_movies.show(10)

# plot actors with top 10 number of movies
top_10_actors = actors_with_top_n_of_movies.limit(10).cache()
sns.set(rc={'figure.figsize':(10,6)})
ax = sns.barplot(x = "primaryName", y = "n_movies", data = top_10_actors.toPandas())
ax.set(title = "Top 10 actors by number of movies", xlabel = "Actor", ylabel = "Number of movies")
ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

## Actors with top 10 number of movies as a worldcloud  
wordcloud_list_of_dicts = top_10_actors.select('primaryName', 'n_movies').rdd.map(lambda row: row.asDict()).collect()
wordcloud_dict = {dict_['primaryName'] : dict_['n_movies'] for dict_ in wordcloud_list_of_dicts}
wordcloud_object = WordCloud(background_color = "white").fit_words(wordcloud_dict)

#displaying the generated image
plt.figure(figsize = (10, 6))
plt.imshow(wordcloud_object)
plt.axis("off")
plt.show()

## Top 10 movies with the biggest number of actors

#construct a table to see movies with the biggest number of actors
query_movies_with_top_n_of_actors = """SELECT primaryTitle, COUNT(*) AS n_actors
                                       FROM movies_and_actors
                                       GROUP BY primaryTitle
                                       ORDER BY n_actors desc
                                       """

movies_with_top_n_of_actors = spark.sql(query_movies_with_top_n_of_actors)
movies_with_top_n_of_actors.createOrReplaceTempView('movies_with_top_n_of_actors')
print('movies_with_top_n_of_actors')
movies_with_top_n_of_actors.show(10)

# plot movies with top 10 number of actors
sns.set(rc={'figure.figsize':(10,6)})
ax = sns.barplot(x = "primaryTitle", y = "n_actors", data = movies_with_top_n_of_actors.limit(10).toPandas())
ax.set(title = "Top 10 movies by number of actors", xlabel = "Movies", ylabel = "Number of actors")
ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

# 3.Apriori Algorithm

## Data organisation

# making a dataframe of movies and actors:  'movie' : ['actor_1', 'actor_2', 'actor_3']
baskets_unqiue_items_df = movies_and_actors.groupBy('primaryTitle').agg(collect_set('primaryName').alias('items'))

# making a list of actors per each movie: [['actor_1', 'actor_2', 'actor_3'],['actor_1', 'actor_2', 'actor_3']]    
list_of_items_per_each_basket = list(baskets_unqiue_items_df.select('items').toPandas()['items'])

# making a tuple of actors per each movie: [('actor_1', 'actor_2', 'actor_3'),('actor_1', 'actor_2', 'actor_3')] 
tuple_of_items_per_each_basket = [tuple(i) for i in list_of_items_per_each_basket]

## Algorithm implementation
from efficient_apriori import apriori

# we choose 100 joint occurences
min_support = 100/len(tuple_of_items_per_each_basket) #has to be expressed as a percentage for efficient-apriori

# we now set min_confidence = 0 to obtain all the rules
min_confidence = 0

# apply apriori algorithm
itemsets, rules = apriori(tuple_of_items_per_each_basket, min_support=min_support, min_confidence=min_confidence)

# 4.Conclusion

## Frequent pairs

# create pandas dataframe with the result(pairs)
actors_pair = np.array([(key,value) for key, value in itemsets[2].items()]).reshape(-1,2)

rules_pair = list(filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules))
conf_pair = np.array([round(rule.confidence,2) for rule in rules_pair]).reshape(-1,2)
lift_pair = np.array([round(rule.lift,2) for rule in rules_pair]).reshape(-1,2)

result_pair_data = np.concatenate((actors_pair, conf_pair, lift_pair), axis=1)

columns_main = ['actors','joint_appearences']
columns_conf = [f'conf_{i}' for i in range(1, conf_pair.shape[1]+1)]
columns_lift = [f'lift_{i}' for i in range(1, lift_pair.shape[1]+1)]
columns_pair = columns_main + columns_conf + columns_lift

result_pair_df = pd.DataFrame(result_pair_data, columns = columns_pair)
result_pair_df= result_pair_df.sort_values('joint_appearences', ascending = False).reset_index(drop = True)

# plot frequent pairs
sns.set(rc={'figure.figsize':(15,6)})
ax = sns.barplot(x = "actors", y = "joint_appearences", data = result_pair_df)
ax.set(title = "Frequent pairs", xlabel = "Actors", ylabel = "Number of movies")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

## Frequent triples

# create pandas dataframe with the result(triples)
actors_triple = np.array([(key,value) for key, value in itemsets[3].items()]).reshape(-1,2)

rules_triple = list(filter(lambda rule: (len(rule.lhs) == 2 and len(rule.rhs) == 1) or (len(rule.lhs) == 1 and len(rule.rhs) == 2) , rules))
conf_triple = np.array([round(rule.confidence,2) for rule in rules_triple]).reshape(-2,6)
lift_triple = np.array([round(rule.lift,2) for rule in rules_triple]).reshape(-2,6)

result_triple_data = np.concatenate((actors_triple, conf_triple, lift_triple), axis=1)

columns_main = ['actors','joint_appearences']
columns_conf = [f'conf_{i}' for i in range(1, conf_triple.shape[1]+1)]
columns_lift = [f'lift_{i}' for i in range(1, lift_triple.shape[1]+1)]
columns_triple = columns_main + columns_conf + columns_lift

result_triple_df = pd.DataFrame(result_triple_data, columns = columns_triple)
result_triple_df= result_triple_df.sort_values('joint_appearences', ascending = False).reset_index(drop = True)
result_triple_df

# plot frequent triples
sns.set(rc={'figure.figsize':(12,6)})
ax = sns.barplot(x = "actors", y = "joint_appearences", data = result_triple_df)
ax.set(title = "Frequent triples", xlabel = "Actors", ylabel = "Number of movies")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)