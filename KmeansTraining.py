
import argparse
import sys
import pyspark
from parallelm.mlops import mlops as pm
from parallelm.mlops import StatCategory as st
from pyspark.sql import SparkSession
from parallelm.mlops.stats.multi_line_graph import MultiLineGraph
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.table import Table
import numpy as np
from pyspark.ml.linalg import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SQLContext, HiveContext, Row, SparkSession
from pyspark.sql.functions import col, count, rand
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer
from pyspark.ml import Pipeline
from jpmml_sparkml import toPMMLBytes
from pyspark.sql.functions import sum, sqrt, min, max

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", help="Data file to use as input")
    parser.add_argument("--with-headers", help="Data file is with headers")
    parser.add_argument("--output-model", help="Path of output model to create")
    parser.add_argument("--K", help="Kmeans number of clusters")
    options = parser.parse_args()
    return options


def eq_dist(x1, y1):
    """
    Calculate Euclidean distance between 2 vectors
    :param x1:
    :param y1:
    :return:
    dist: distance between vectors
    """
    x = np.asarray(x1)
    y = np.asarray(y1)
    len_x = len(x)
    len_y = len(y)
    if not len_x == len_y:
        print("error: elements not of the same shape")
        dist = 0
    else:
        dist1 = np.sum((x-y)**2)
        if dist1 > 0:
            dist = np.sqrt(dist1)
        else:
            dist = 0
    return float(dist)


def save_model_locally(pmml_file, pm_options):
    """
    The function saves the model to the provided path
    :param pmml_file: model to save
    :param pm_options: options to get the file path
    :return:
    """
    print("PM: pmml model:\n{}".format(pmml_file))
    output_file = open(pm_options.output_model, "w+")
    output_file.write(pmml_file)
    output_file.close()
    return


def kmeans_train(pm_options, spark):
    """
    Kmeans Training function
    :param pm_options:
    :param spark:
    :return:
    """

    # Import Data
    ##################################
    input_data = (spark.read.format("csv")
                  .option("header", pm_options.with_headers)
                  .option("ignoreLeadingWhiteSpace", "true")
                  .option("ignoreTrailingWhiteSpace", "true")
                  .option("inferschema", "true")
                  .load(pm_options.data_file)).repartition(10)

    column_names_all = input_data.columns
    if not pm_options.with_headers == "true":
        for col_index in range(0, len(column_names_all)):
            input_data = input_data.withColumnRenamed(column_names_all[col_index],
                                                      'c' + str(col_index))

    input_data = input_data.cache()

    input_train = input_data
    input_test = input_data

    # SparkML pipeline
    ##################################
    exclude_cols = []
    column_names = input_train.columns
    input_col_names = []
    for elmts in column_names:
        ind = True
        for excludes in exclude_cols:
            if elmts == excludes:
                ind = False
        if ind:
            input_col_names.append(elmts)
    print(input_col_names)

    vector_assembler = VectorAssembler(
        inputCols=input_col_names,
        outputCol="features")
    kmeans_pipe = KMeans(
        k=int(pm_options.K),
        initMode="k-means||",
        initSteps=2,
        tol=1e-4,
        maxIter=100,
        featuresCol="features")
    full_pipe = [vector_assembler, kmeans_pipe]
    model_kmeans = Pipeline(stages=full_pipe).fit(input_train)

    # Test validation and statistics collection
    ############################################################
    predicted_df = model_kmeans.transform(input_test)

    print("model_kmeans.stages(1) = ", model_kmeans.stages[1])

    sum_errors = model_kmeans.stages[1].computeCost(predicted_df)
    print("Sum of Errors for Kmeans = " + str(sum_errors))

    # Shows the result.
    kmeans_centers = model_kmeans.stages[1].clusterCenters()
    print("Kmeans Centers: ")
    for center in kmeans_centers:
        print(center)

    # calculating stats
    ############################################################

    # Calculating Inter cluster distance
    inter_cluster_distance = np.zeros((len(kmeans_centers), len(kmeans_centers)))

    for centerIndex1 in range(0, len(kmeans_centers)):
        for centerIndex2 in range(0, len(kmeans_centers)):
            inter_cluster_distance[centerIndex1, centerIndex2] =\
                eq_dist(kmeans_centers[centerIndex1], kmeans_centers[centerIndex2])

    print("inter_cluster_distance = ", inter_cluster_distance)
    # Calculating Intra cluster distances and the bars for the cluster distribution
    intra_cluster_distance = np.zeros(len(kmeans_centers))
    cluster_dist = np.zeros(len(kmeans_centers))

    for centerIndex1 in range(0, len(kmeans_centers)):
        filtered_df = predicted_df.filter(predicted_df["prediction"] == centerIndex1)
        cluster_dist[centerIndex1] = filtered_df.count()
        if cluster_dist[centerIndex1] == 0:
            intra_cluster_distance[centerIndex1] = 0
        else:
            filtered_df =\
                filtered_df.withColumn('distance',
                                       udf(eq_dist, FloatType())(col("features"),
                                            array([lit(v) for v in kmeans_centers[centerIndex1]])))
            intra_cluster_distance[centerIndex1] =\
                filtered_df.agg(sum("distance")).first()[0] / cluster_dist[centerIndex1]

    # calculating Davis-Boulding Index
    ############################################################
    # R[i,j] = (S[i] + S[j])/M[i,j]
    # D[i] = max(R[i,j]) for i !=j
    # DB = (1/K) * sum(D[i])
    r_index = np.zeros((len(kmeans_centers), len(kmeans_centers)))
    for centerIndex1 in range(0, len(kmeans_centers)):
        for centerIndex2 in range(0, len(kmeans_centers)):
            r_index[centerIndex1, centerIndex2] = 0
            if not inter_cluster_distance[centerIndex1, centerIndex2] == 0:
                r_index[centerIndex1, centerIndex2] =\
                    (intra_cluster_distance[centerIndex1] + intra_cluster_distance[centerIndex2])\
                    / inter_cluster_distance[centerIndex1, centerIndex2]
    d_index = np.max(r_index, axis=0)
    db_index = np.sum(d_index, axis=0) / len(kmeans_centers)

    # pmml model generation
    ############################################################
    pmml_file = toPMMLBytes(spark, input_train, model_kmeans).decode("UTF-8")

    # PM stats
    ############################################################
    print("Sum of Errors for Kmeans = " + str(sum_errors))
    pm.set_stat("Sum of Errors for Kmeans", sum_errors, st.TIME_SERIES)

    print("Davies-Bouldin index = " + str(db_index))
    pm.set_stat("Davies-Bouldin index", db_index, st.TIME_SERIES)

    # Tables
    tbl_col_name = []
    for j in range(0, len(kmeans_centers)):
        tbl_col_name.append(str(j))
    tbl = Table().name("Inter cluster distance").cols(tbl_col_name)
    for j in range(0, len(kmeans_centers)):
        tbl.add_row(str(j) + ":", ["%.2f" % x for x in inter_cluster_distance[j, :]])
    pm.set_stat(tbl)

    tbl = Table().name("Intra cluster avg. distance").cols(tbl_col_name)
    tbl.add_row("Distances:", ["%.2f" % x for x in intra_cluster_distance])
    pm.set_stat(tbl)

    tbl_col_name1 = []
    for j in range(0, len(kmeans_centers[0])):
        tbl_col_name1.append(str(j))
    tbl = Table().name("Centers (for K<6, Attr<11)").cols(tbl_col_name1)
    for j in range(0, len(kmeans_centers)):
        tbl.add_row("center" + str(j) + ":", ["%.2f" % x for x in kmeans_centers[j]])
    pm.set_stat(tbl)

    # BarGraph
    bar = BarGraph().name("Cluster Destribution").cols(tbl_col_name).data(cluster_dist.tolist())
    pm.stat(bar)

    print("PM: generating histogram from data-frame and model")
    print("PM:" + pmml_file)
    try:
        pm.set_data_distribution_stat(data=input_train, model=pmml_file)
        print("PM: done generating histogram")
    except Exception as e:
        print("PM: failed to generate histogram using pm.stat")
        print(e)

    return pmml_file


def main():
    options = parse_args()
    print("PM: Configuration:")
    print("PM: Data file:            {}".format(options.data_file))
    print("PM: Output model:         {}".format(options.output_model))

    print()
    print("PM: Starting  code")
    print()

    print("PM: imported function!")
    print("PM: creating spark session")
    spark = SparkSession.builder.appName("KmeansTrain").getOrCreate()

    print("PM: calling pm.init()")
    pm.init(spark.sparkContext)

    print("PM: calling Kmeans_train")

    # Taking out the pmml model so jpmml evaluator can use this
    pmml_file = kmeans_train(pm_options=options, spark=spark)
    print("PM: json returned from Kmeans_train function!")

    print("PM: Saving model")
    save_model_locally(pmml_file, options)
    print("PM: model file saved locally!")

    print("PM: calling spark.stop")
    spark.stop()

    print("PM: calling pm.done()")
    pm.done()
    print("PM: after pm.done")


if __name__ == "__main__":
    main()
