# Databricks notebook source
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

train = spark.sql("SELECT Id, SalePrice as label, Neighborhood, LotArea, OverallQual, OverallCond, YearBuilt, TotalBsmtSF, GrLivArea, MoSold, YrSold FROM raw_trainHP")
display(train.select("*"))

# COMMAND ----------

test.describe().show()

# COMMAND ----------

test = spark.sql("SELECT Id, Neighborhood, LotArea, OverallQual, OverallCond, YearBuilt, TotalBsmtSF, GrLivArea, MoSold, YrSold FROM raw_testHP")
display(test.select("*"))

# COMMAND ----------

test = test.withColumn("TotalBsmtSF", col("TotalBsmtSF").cast(IntegerType()))
test.printSchema

# COMMAND ----------

nullnhood = spark.sql("SELECT * from raw_testHP where Neighborhood is NULL")

# COMMAND ----------

from pyspark.sql import functions as F
test.filter(F.isnull("YrSold")).show()

# COMMAND ----------

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
lr = LinearRegression(maxIter=10)

# COMMAND ----------

#Define Pipeline
pipeline = Pipeline(stages=[Neighborhood_indexer, YearBuilt_indexer, MoSold_indexer, YrSold_indexer, assembler, lr])

# COMMAND ----------

ind_model = pipeline.fit(train)
train_final = ind_model.transform(test)
display(train_final)

# COMMAND ----------

# Fit the model
lrModel = lr.fit(train_final)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
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

