# Databricks notebook source
# First Submission Date = June 17, 2018
# Test RMSE = 0.85055
# Place = 4968
# Only 1 tuned hyper parameter - regParam

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# MAGIC %sql
# MAGIC use global_solutions_fs

# COMMAND ----------

# Import Training and Test Data
train = spark.sql("SELECT Id, SalePrice as label, Neighborhood, LotArea, OverallQual, OverallCond, YearBuilt, TotalBsmtSF, GrLivArea, MoSold, YrSold FROM raw_trainHP")
test = spark.sql("SELECT Id, Neighborhood, LotArea, OverallQual, OverallCond, YearBuilt, TotalBsmtSF, GrLivArea, MoSold, YrSold FROM raw_testHP")

# COMMAND ----------

#Convert TotalBsmtSF Column to Int
test = test.withColumn("TotalBsmtSF", col("TotalBsmtSF").cast(IntegerType()))

# Fill 1 Null TotalBsmtSF Row with Average 
test = test.na.fill({"TotalBsmtSF": 1046})

# COMMAND ----------

#Define Indexers
Neighborhood_indexer = StringIndexer(inputCol="Neighborhood", outputCol="Neighborhood_Indexed", handleInvalid="keep")
YearBuilt_indexer = StringIndexer(inputCol="YearBuilt", outputCol="YearBuilt_Indexed", handleInvalid="keep")
MoSold_indexer = StringIndexer(inputCol="MoSold", outputCol="MoSold_Indexed", handleInvalid="keep")
YrSold_indexer = StringIndexer(inputCol="YrSold", outputCol="YrSold_Indexed", handleInvalid="keep")

#Define Vector Assembler
assembler = VectorAssembler(
    inputCols=["Neighborhood_Indexed", "YearBuilt_Indexed", "MoSold_Indexed", "YrSold_Indexed", "GrLivArea", "TotalBsmtSF", "LotArea", "OverallQual", "OverallCond"],
    outputCol="features")

#Define Linear Regression Model
lr = LinearRegression(maxIter=100)

# COMMAND ----------

#Define Pipeline
pipeline = Pipeline(stages=[Neighborhood_indexer, YearBuilt_indexer, MoSold_indexer, YrSold_indexer, assembler, lr])

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.05, 0.01])\
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0]).build()
evaluator = RegressionEvaluator(metricName="rmse", labelCol="label")
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 
cvModel = crossval.fit(train)

# COMMAND ----------

prediction = cvModel.transform(test)

# COMMAND ----------

display(prediction.selectExpr("id as  Id", "prediction as SalePrice"))

# COMMAND ----------

