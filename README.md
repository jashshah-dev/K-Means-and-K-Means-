# K-Means-and-K-means-plus-plus




This repository provides a Python implementation of the K-means and K-means++ clustering algorithms from scratch. The code allows for the comparison of K-means and K-means++ algorithms, as well as the evaluation of their performance in clustering analysis. The analysis is conducted using the 130 Hospitals Diabetic Dataset from the US.

## Contents

- [Background](#background)
- [Features](#features)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [Comparison Metric Used](#Comparison-Metric-Used)
- [Results](#results)


## Background

This project focuses on the implementation and comparison of the K-means and K-means++ clustering algorithms. K-means is a popular unsupervised learning algorithm used for partitioning a dataset into K distinct clusters. It aims to minimize the within-cluster sum of squares, where each data point is assigned to the cluster with the closest mean.

K-means++ is an enhancement over the standard K-means algorithm that addresses one of its limitations—the sensitivity to the initial centroid selection. In the original K-means algorithm, the initial centroids are randomly chosen, which can result in suboptimal clustering outcomes. K-means++ introduces a smarter initialization step by selecting initial centroids that are distant from each other. This improves the chances of converging to a better solution and reduces the sensitivity to the initial starting point.

In this project, we provide a Python implementation of both the K-means and K-means++ algorithms from scratch. By comparing the results of these algorithms, we gain insights into the effectiveness of the K-means++ initialization technique in improving clustering performance. We utilize the 130 Hospitals Diabetic Dataset from the US to evaluate and analyze the clustering outcomes.

The 130 Hospitals Diabetic Dataset is a comprehensive dataset that includes various attributes related to diabetic patients, such as patient demographics, medical history, treatment details, and hospital-specific data. By applying the K-means and K-means++ algorithms to this dataset, we aim to discover inherent structures and patterns among the patients, potentially leading to valuable insights for personalized healthcare and treatment strategies.

The K-means algorithm iteratively updates the cluster centroids until convergence, assigning data points to the nearest centroid. On the other hand, K-means++ incorporates a more intelligent initialization step, improving the overall quality of the clustering. By comparing the clustering results between the two algorithms, we can evaluate the impact of the initialization process on the final clustering outcome.

Through this project, we aim to provide a clear understanding of the K-means and K-means++ algorithms, highlight the benefits of using K-means++ initialization, and demonstrate their applications in real-world datasets. By offering a Python implementation from scratch, we provide an opportunity for researchers and practitioners to gain hands-on experience with these clustering algorithms and customize them according to their specific requirements.

## K-means Algorithm
The K-means algorithm is an iterative clustering algorithm that aims to partition a given dataset into K distinct clusters. The algorithm starts by randomly selecting K initial cluster centroids and assigns each data point to the nearest centroid. It then recalculates the centroid positions based on the assigned data points and repeats the process until convergence. The final result is a set of K clusters, each represented by its centroid.

While K-means is relatively simple and efficient, it has some limitations. One major drawback is its sensitivity to the initial centroid positions, which can lead to suboptimal clustering results. Additionally, it assumes that the clusters are spherical and have similar sizes, which may not always hold in real-world datasets.


## K-means++ Algorithm
K-means++ is an extension of the K-means algorithm that addresses the issue of selecting good initial centroids. The K-means++ algorithm introduces a more intelligent initialization step to improve the chances of finding a globally optimal solution.

Instead of randomly selecting the initial centroids, K-means++ follows a probabilistic approach. It starts by selecting the first centroid uniformly at random from the data points. Subsequent centroids are selected based on their distance from the already chosen centroids, with higher probabilities given to data points that are farther away from existing centroids. This initialization process reduces the likelihood of getting stuck in local optima and typically leads to better clustering results.

The K-means++ algorithm retains the iterative update step of K-means, where the centroids are recomputed based on the assigned data points. By combining smart initialization with the iterative optimization process, K-means++ tends to converge faster and provide more accurate cluster assignments compared to the standard K-means algorithm.


## Features

Implementation of K-means and K-means++ clustering algorithms from scratch

Comparison of clustering results between K-means and K-means++ algorithms

Utilization of the 130 Hospitals Diabetic Dataset for analysis and evaluation

## Dataset

The dataset used in this project is the 130 Hospitals Diabetic Dataset from the US. It contains information on diabetic patients, including patient demographics, medical history, and treatment details. The dataset provides a rich source of information for clustering analysis and algorithm comparison.


## Comparison Metrics Used

## Comparison Metrics
In order to assess the performance of different clustering algorithms, we utilize several comparison metrics that provide insights into the quality and characteristics of the obtained clusters. The following metrics are employed in our evaluation:

## Calinski-Harabasz Index
The Calinski-Harabasz index, also known as the Variance Ratio Criterion, measures the ratio of between-cluster dispersion to within-cluster dispersion. A higher index value indicates better-defined and well-separated clusters. It is calculated based on the dispersion of data points around their respective cluster centroids and the dispersion between different cluster centroids.

## Silhouette Score
The Silhouette score evaluates the quality of clustering by measuring the cohesion and separation of data points within and between clusters. It computes the average Silhouette coefficient for all data points, ranging from -1 to 1. A higher Silhouette score suggests that the clusters are well-separated and internally cohesive, while a lower score indicates overlapping or poorly separated clusters.

## Within Sum of Square Error (SSE)
The Within Sum of Square Error, also known as the inertia or distortion, quantifies the compactness of clusters. It measures the sum of squared distances between each data point and its assigned cluster centroid. A lower SSE value indicates that the data points within each cluster are closer to their respective centroids, indicating better cluster compactness.

## Davies-Bouldin Score
The Davies-Bouldin score assesses the clustering quality by considering both the within-cluster dispersion and the between-cluster separation. It measures the average similarity between each cluster and its most similar cluster, while also considering the cluster sizes. A lower Davies-Bouldin score indicates better-defined and well-separated clusters.

## Results
![image](https://github.com/jashshah-dev/K-Means-and-K-means-plus-plus/assets/132673402/363711e4-f4b6-4ff5-8d7c-9e3de84d1131)













