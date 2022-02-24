# Market-Basket-Analysis

This project is supposed to demonstrate an approach of solving a classical class of problems called Market Basket analyses. The main idea behind this problem – is to find items that are frequently
bought together in one consumer basket using efficient computational algorithms for processing massive data(>1G). The naïve approach of cycling over each item and each available itemset is a memory expensive task and could probably be not executed on a standard machine. In order to accomplish the computation using limited resources we demonstrate a MapReduce approach that allows us to find the most frequent pairs of actors appearing together on screen using IMDB dataset.

## **Implemented functionality**

1. Data downloading via Kaggle API
2. Exploratory Data analysis
3. Finding insights
4. Apriori algorithm implementation
5. Finding most frequent pairs and triples of actors playing together in a movie
