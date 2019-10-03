from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
import pandas as pd
import sys
from time import time
import csv

conf = SparkConf().setAppName("HAR").setMaster(master) # Comment to run in Jupyter
sc = SparkContext(conf=conf)                           # Comment to run in Jupyter

res = open("wisdm_main_ver_0.0/main_result/result.txt", 'w+')
sys.stdout = res

sqlContext = SQLContext(sc)

print("Loading Data Set...")
# Data Ingestion and Extraction
data = sqlContext.read.format('com.databricks.spark.csv').                 \
                                options(header='true', inferschema='true').\
                                load('wisdm_main_ver_0.0/data/wisdm_data.csv')

drop_list = ['USER','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9',\
            'Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9',        \
            'Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9']

data = data.select([column for column in data.columns if column not in drop_list])
print("Data Schema------------------------------------------------------------")
data.printSchema()
print("Sample Data------------------------------------------------------------")
data.show(5)

# SQL Injection
from pyspark.sql.functions import col
print("Activity Count----------------------------------------------------------")
data.groupBy("activity")          \
    .count()                      \
    .orderBy(col("count").desc()) \
    .show()

pd.DataFrame(data.take(10), columns=data.columns).transpose()
numeric_features = [t[0] for t in data.dtypes if (t[1] == 'double' or t[1]=='int')]
print("Summary---------------------------------------------------------------")
print(data.select(numeric_features).describe().toPandas().transpose())

# Creating Dataframe
cols = data.columns
df = data.select(cols)

print("\n===========================MODELING PIPELINE==============================\n")
# Model Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
categoricalColumns = ['XPEAK','YPEAK','ZPEAK']
stages = []

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'ACTIVITY', outputCol = 'label')
stages += [label_stringIdx]

numericCols = ['XAVG','YAVG','ZAVG','XABSDEV','YABSDEV','ZABSDEV','XSTDDEV','YSTDDEV','ZSTDDEV','RESULTANT']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
print("Model Pipeline Schema------------------------------------------------------------")
df.printSchema()
print("Sample Feature Data------------------------------------------------------------")
print(pd.DataFrame(df.take(5), columns=df.columns))

# Partition Training & Test sets
train, test = df.randomSplit([0.7, 0.3], seed = 2018)
#train.toPandas().to_csv('wisdm_main_ver_0.0/train/train.csv')
#test.toPandas().to_csv('wisdm_main_ver_0.0/test/test.csv')
print("\n===========================TRAINING AND TESTING==============================\n")
print("Training Dataset Count : " + str(train.count()))
print("Test Dataset Count     : " + str(test.count()))

minimized_view = ['XPEAK','YPEAK','ZPEAK','XABSDEV','YABSDEV','ZABSDEV']

train.select([column for column in test.columns if column not in minimized_view]).show(5)
test.select([column for column in test.columns if column not in minimized_view]).show(5)

# Modifying Test 
skipped    = ['XPEAK','YPEAK','ZPEAK',      \
              'XAVG','YAVG','ZAVG',         \
              'XABSDEV','YABSDEV','ZABSDEV',\
              'XSTDDEV','YSTDDEV','ZSTDDEV',\
              'RESULTANT','ACTIVITY']
test_data  = test.select([column for column in test.columns if column not in skipped])
#test_data.toPandas().to_csv(r'wisdm_main_ver_0.0/test/test_data_minimal.csv')
test_data.show(5)

print("============================CLASSIFICATION AND EVALUATION============================")
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

#******************************************************************
# Logistic Regression
#******************************************************************
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
t0 = time()
lrModel = lr.fit(train)
lRtt = round((time()-t0),3)
print(lrModel)
print ("Classifier trained in %g seconds"%lRtt)
t0 = time()
predictions = lrModel.transform(test_data)
lRst = round((time()-t0),3)
print ("Prediction made in %g seconds"%lRst)


predictions.filter(predictions['prediction'] == 5) \
    .select("UID","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 5, truncate = 30)

# Binary Classification Evaluator
print("\n-----------Binary Classification Evaluator-------------\n")

evaluator = BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction")
lRRaw = evaluator.evaluate(predictions)
print("Binary Classifier Raw Prediction ------------: %g"%lRRaw)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderPR")
lRAuPR = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under PR --------------: %g"%lRAuPR)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderROC")
lRAuROC = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under ROC -------------: %g"%lRAuROC)

# MultiClass Classification Evaluator
print("\n-----------MultiClass Classification Evaluaton---------\n")

evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
lRf1=evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print("MultiClass F1 -------------------------------: %g"%lRf1)
lRwP=evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print("MultiClass Weighted Precision ---------------: %g"%lRwP)
lRwR=evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print("MultiClass Weighted Recall ------------------: %g"%lRwR)
lRaccuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print("MultiClass Accuracy -------------------------: %g"%lRaccuracy)

# Regression Evaluator
print("\n----------------Regression Evaluator-------------------\n")

#metric name in evaluation - one of:
#rmse - root mean squared error (default)
#mse - mean squared error
#r2 - r^2 metric
#mae - mean absolute error.
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
lRrmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data -: %g" % lRrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")
lRmse = evaluator.evaluate(predictions)
print("Mean Squared Error on test data -------------: %g" % lRrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
lRr2 = evaluator.evaluate(predictions)
print("R^2 metric on test data ---------------------: %g" % lRr2)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
lRmae = evaluator.evaluate(predictions)
print("Mean Absolute Error on test data ------------: %g" % lRmae)

# Additional Param
print("\n------------------Additional Factors--------------------\n")
lp = predictions.select( "label", "prediction")
lRcountTotal = predictions.count()

print("Total Count          = %g"%lRcountTotal)

lRcorrect=lp.filter(col('label')== col('prediction')).count()
print("Total Correct        = %g"%lRcorrect)
lRwrong = lp.filter(~(col('label') == col('prediction'))).count()
print("Total Wrong          = %g"%lRwrong)

lRratioWrong=float(float(lRwrong)/float(lRcountTotal))
print("Wrong Ratio          = %g"%lRratioWrong)
lRratioCorrect=float(float(lRcorrect)/float(lRcountTotal))
print("Right Ratio          = %g"%lRratioCorrect)
print("\n*********************************************************\n")

# Cross Validator 
#***************************************************************************
# Create 5-fold CrossValidator for Logistic Regression
#***************************************************************************
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
            .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
            .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
#            .addGrid(model.maxIter, [10]) #Number of iterations
#            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())

cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)

t0 = time()
cvModel = cv.fit(train)
lRcFtt = round((time()-t0),3)
print(str(cvModel)+" for Logistic Regression")
print ("Classifier trained in %g seconds"%lRcFtt)
t0 = time()
predictions = cvModel.transform(test_data)
lRcFst = round((time()-t0),3)
print ("Prediction made in %g seconds"%lRcFst)

predictions.filter(predictions['prediction'] == 0) \
    .select("UID","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 5, truncate = 30)

# Binary Classification Evaluator
print("\n-----------Binary Classification Evaluator-------------\n")

evaluator = BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction")
lRcFRaw = evaluator.evaluate(predictions)
print("Binary Classifier Raw Prediction ------------: %g"%lRcFRaw)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderPR")
lRcFAuPR = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under PR --------------: %g"%lRcFAuPR)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderROC")
lRcFAuROC = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under ROC -------------: %g"%lRcFAuROC)

# MultiClass Classification Evaluator
print("\n-----------MultiClass Classification Evaluaton---------\n")

evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
lRcFf1=evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print("MultiClass F1 -------------------------------: %g"%lRcFf1)
lRcFwP=evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print("MultiClass Weighted Precision ---------------: %g"%lRcFwP)
lRcFwR=evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print("MultiClass Weighted Recall ------------------: %g"%lRcFwR)
lRcFaccuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print("MultiClass Accuracy -------------------------: %g"%lRcFaccuracy)

# Regression Evaluator
print("\n----------------Regression Evaluator-------------------\n")

#metric name in evaluation - one of:
#rmse - root mean squared error (default)
#mse - mean squared error
#r2 - r^2 metric
#mae - mean absolute error.
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
lRcFrmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data -: %g" % lRcFrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")
lRcFmse = evaluator.evaluate(predictions)
print("Mean Squared Error on test data -------------: %g" % lRcFrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
lRcFr2 = evaluator.evaluate(predictions)
print("R^2 metric on test data ---------------------: %g" % lRcFr2)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
lRcFmae = evaluator.evaluate(predictions)
print("Mean Absolute Error on test data ------------: %g" % lRcFmae)

# Additional Param
print("\n------------------Additional Factors--------------------\n")
lp = predictions.select( "label", "prediction")
lRcFcountTotal = predictions.count()

print("Total Count          = %g"%lRcFcountTotal)

lRcFcorrect=lp.filter(col('label')== col('prediction')).count()
print("Total Correct        = %g"%lRcFcorrect)
lRcFwrong = lp.filter(~(col('label') == col('prediction'))).count()
print("Total Wrong          = %g"%lRcFwrong)

lRcFratioWrong=float(float(lRcFwrong)/float(lRcFcountTotal))
print("Wrong Ratio          = %g"%lRcFratioWrong)
lRcFratioCorrect=float(float(lRcFcorrect)/float(lRcFcountTotal))
print("Right Ratio          = %g"%lRcFratioCorrect)
print("\n*********************************************************\n")

#*****************************************************************
# Decision Tree Classifier
#*****************************************************************
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)

t0 = time()
dtModel = dt.fit(train)
dTtt = round((time()-t0),3)
print(dtModel)
print ("Classifier trained in %g seconds"%dTtt)
t0 = time()
predictions = dtModel.transform(test_data)
dTst = round((time()-t0),3)
print ("Prediction made in %g seconds"%dTst)

predictions.filter(predictions['prediction'] == 0) \
    .select("UID","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 5, truncate = 30)

# Binary Classification Evaluator
print("\n-----------Binary Classification Evaluator-------------\n")

evaluator = BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction")
dTRaw = evaluator.evaluate(predictions)
print("Binary Classifier Raw Prediction ------------: %g"%dTRaw)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderPR")
dTAuPR = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under PR --------------: %g"%dTAuPR)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderROC")
dTAuROC = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under ROC -------------: %g"%dTAuROC)

# MultiClass Classification Evaluator
print("\n-----------MultiClass Classification Evaluaton---------\n")

evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
dTf1=evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print("MultiClass F1 -------------------------------: %g"%dTf1)
dTwP=evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print("MultiClass Weighted Precision ---------------: %g"%dTwP)
dTwR=evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print("MultiClass Weighted Recall ------------------: %g"%dTwR)
dTaccuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print("MultiClass Accuracy -------------------------: %g"%dTaccuracy)

# Regression Evaluator
print("\n----------------Regression Evaluator-------------------\n")

#metric name in evaluation - one of:
#rmse - root mean squared error (default)
#mse - mean squared error
#r2 - r^2 metric
#mae - mean absolute error.
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
dTrmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data -: %g" % dTrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")
dTmse = evaluator.evaluate(predictions)
print("Mean Squared Error on test data -------------: %g" % dTrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
dTr2 = evaluator.evaluate(predictions)
print("R^2 metric on test data ---------------------: %g" % dTr2)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
dTmae = evaluator.evaluate(predictions)
print("Mean Absolute Error on test data ------------: %g" % dTmae)

# Additional Param
print("\n------------------Additional Factors--------------------\n")
lp = predictions.select( "label", "prediction")
dTcountTotal = predictions.count()

print("Total Count          = %g"%dTcountTotal)

dTcorrect=lp.filter(col('label')== col('prediction')).count()
print("Total Correct        = %g"%dTcorrect)
dTwrong = lp.filter(~(col('label') == col('prediction'))).count()
print("Total Wrong          = %g"%dTwrong)

dTratioWrong=float(float(dTwrong)/float(dTcountTotal))
print("Wrong Ratio          = %g"%dTratioWrong)
dTratioCorrect=float(float(dTcorrect)/float(dTcountTotal))
print("Right Ratio          = %g"%dTratioCorrect)
print("\n*********************************************************\n")

#*********************************************************************
# Create 5-fold CrossValidator for Decision Tree
#*********************************************************************
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
#            .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
#            .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
#            .addGrid(model.maxIter, [10]) #Number of iterations
#            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())
cv = CrossValidator(estimator=dt, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)

t0 = time()
cvModel = cv.fit(train)
dTcFtt = round((time()-t0),3)
print(str(cvModel)+" for Decision Tree")
print ("Classifier trained in %g seconds"%dTcFtt)
t0 = time()
predictions = cvModel.transform(test_data)
dTcFst = round((time()-t0),3)
print ("Prediction made in %g seconds"%dTcFst)

predictions.filter(predictions['prediction'] == 0) \
    .select("UID","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 5, truncate = 30)

# Binary Classification Evaluator
print("\n-----------Binary Classification Evaluator-------------\n")

evaluator = BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction")
dTcFRaw = evaluator.evaluate(predictions)
print("Binary Classifier Raw Prediction ------------: %g"%dTcFRaw)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderPR")
dTcFAuPR = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under PR --------------: %g"%dTcFAuPR)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderROC")
dTcFAuROC = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under ROC -------------: %g"%dTcFAuROC)

# MultiClass Classification Evaluator
print("\n-----------MultiClass Classification Evaluaton---------\n")

evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
dTcFf1=evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print("MultiClass F1 -------------------------------: %g"%dTcFf1)
dTcFwP=evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print("MultiClass Weighted Precision ---------------: %g"%dTcFwP)
dTcFwR=evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print("MultiClass Weighted Recall ------------------: %g"%dTcFwR)
dTcFaccuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print("MultiClass Accuracy -------------------------: %g"%dTcFaccuracy)

# Regression Evaluator
print("\n----------------Regression Evaluator-------------------\n")

#metric name in evaluation - one of:
#rmse - root mean squared error (default)
#mse - mean squared error
#r2 - r^2 metric
#mae - mean absolute error.
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
dTcFrmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data -: %g" % dTcFrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")
dTcFmse = evaluator.evaluate(predictions)
print("Mean Squared Error on test data -------------: %g" % dTcFrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
dTcFr2 = evaluator.evaluate(predictions)
print("R^2 metric on test data ---------------------: %g" % dTcFr2)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
dTcFmae = evaluator.evaluate(predictions)
print("Mean Absolute Error on test data ------------: %g" % dTcFmae)

# Additional Param
print("\n------------------Additional Factors--------------------\n")
lp = predictions.select( "label", "prediction")
dTcFcountTotal = predictions.count()

print("Total Count          = %g"%dTcFcountTotal)

dTcFcorrect=lp.filter(col('label')== col('prediction')).count()
print("Total Correct        = %g"%dTcFcorrect)
dTcFwrong = lp.filter(~(col('label') == col('prediction'))).count()
print("Total Wrong          = %g"%dTcFwrong)

dTcFratioWrong=float(float(dTcFwrong)/float(dTcFcountTotal))
print("Wrong Ratio          = %g"%lRcFratioWrong)
dTcFratioCorrect=float(float(dTcFcorrect)/float(dTcFcountTotal))
print("Right Ratio          = %g"%dTcFratioCorrect)
print("\n*********************************************************\n")

#*****************************************************************
# Random Forest Classifier
#*****************************************************************

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label',numTrees = 100,maxDepth = 4,maxBins = 32)

t0 = time()
rfModel = rf.fit(train)
rFtt = round((time()-t0),3)
print(rfModel)
print ("Classifier trained in %g seconds"%rFtt)
t0 = time()
predictions = rfModel.transform(test_data)
rFst = round((time()-t0),3)
print ("Prediction made in %g seconds"%rFst)

predictions.filter(predictions['prediction'] == 0) \
    .select("UID","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 5, truncate = 30)

# Binary Classification Evaluator
print("\n-----------Binary Classification Evaluator-------------\n")

evaluator = BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction")
rFRaw = evaluator.evaluate(predictions)
print("Binary Classifier Raw Prediction ------------: %g"%rFRaw)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderPR")
rFAuPR = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under PR --------------: %g"%rFAuPR)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderROC")
rFAuROC = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under ROC -------------: %g"%rFAuROC)

# MultiClass Classification Evaluator
print("\n-----------MultiClass Classification Evaluaton---------\n")

evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
rFf1=evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print("MultiClass F1 -------------------------------: %g"%rFf1)
rFwP=evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print("MultiClass Weighted Precision ---------------: %g"%rFwP)
rFwR=evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print("MultiClass Weighted Recall ------------------: %g"%rFwR)
rFaccuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print("MultiClass Accuracy -------------------------: %g"%rFaccuracy)

# Regression Evaluator
print("\n----------------Regression Evaluator-------------------\n")

#metric name in evaluation - one of:
#rmse - root mean squared error (default)
#mse - mean squared error
#r2 - r^2 metric
#mae - mean absolute error.
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rFrmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data -: %g" % rFrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")
rFmse = evaluator.evaluate(predictions)
print("Mean Squared Error on test data -------------: %g" % rFrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
rFr2 = evaluator.evaluate(predictions)
print("R^2 metric on test data ---------------------: %g" % rFr2)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
rFmae = evaluator.evaluate(predictions)
print("Mean Absolute Error on test data ------------: %g" % rFmae)

# Additional Param
print("\n------------------Additional Factors--------------------\n")
lp = predictions.select( "label", "prediction")
rFcountTotal = predictions.count()

print("Total Count          = %g"%rFcountTotal)

rFcorrect=lp.filter(col('label')== col('prediction')).count()
print("Total Correct        = %g"%rFcorrect)
rFwrong = lp.filter(~(col('label') == col('prediction'))).count()
print("Total Wrong          = %g"%rFwrong)

rFratioWrong=float(float(rFwrong)/float(rFcountTotal))
print("Wrong Ratio          = %g"%rFratioWrong)
rFratioCorrect=float(float(rFcorrect)/float(rFcountTotal))
print("Right Ratio          = %g"%rFratioCorrect)
print("\n*********************************************************\n")

#*********************************************************************
# Create 5-fold CrossValidator for Random Forest
#*********************************************************************
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
#            .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
#            .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
#            .addGrid(model.maxIter, [10]) #Number of iterations
#            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())
cv = CrossValidator(estimator=rf, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)

t0 = time()
cvModel = cv.fit(train)
rFcFtt = round((time()-t0),3)
print(str(cvModel)+" for Random Forest")
print ("Classifier trained in %g seconds"%rFcFtt)
t0 = time()
predictions = cvModel.transform(test_data)
rFcFst = round((time()-t0),3)
print ("Prediction made in %g seconds"%rFcFst)


predictions.filter(predictions['prediction'] == 0) \
    .select("UID","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 5, truncate = 30)

# Binary Classification Evaluator
print("\n-----------Binary Classification Evaluator-------------\n")

evaluator = BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction")
rFcFraw = evaluator.evaluate(predictions)
print("Binary Classifier Raw Prediction ------------: %g"%rFcFraw)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderPR")
rFcFauPR = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under PR --------------: %g"%rFcFauPR)
evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderROC")
rFcFauROC = evaluator.evaluate(predictions)
print("Binary Clasifier Area Under ROC -------------: %g"%rFcFauROC)

# MultiClass Classification Evaluator
print("\n-----------MultiClass Classification Evaluaton---------\n")

evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
rFcFf1=evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print("MultiClass F1 -------------------------------: %g"%rFcFf1)
rFcFwP=evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print("MultiClass Weighted Precision ---------------: %g"%rFcFwP)
rFcFwR=evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print("MultiClass Weighted Recall ------------------: %g"%rFcFwR)
rFcFaccuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print("MultiClass Accuracy -------------------------: %g"%rFcFaccuracy)

# Regression Evaluator
print("\n----------------Regression Evaluator-------------------\n")

#metric name in evaluation - one of:
#rmse - root mean squared error (default)
#mse - mean squared error
#r2 - r^2 metric
#mae - mean absolute error.
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rFcFrmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data -: %g" % rFcFrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")
rFcFmse = evaluator.evaluate(predictions)
print("Mean Squared Error on test data -------------: %g" % rFcFrmse)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
rFcFr2 = evaluator.evaluate(predictions)
print("R^2 metric on test data ---------------------: %g" % rFcFr2)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
rFcFmae = evaluator.evaluate(predictions)
print("Mean Absolute Error on test data ------------: %g" % rFcFmae)

# Additional Param
print("\n------------------Additional Factors--------------------\n")
lp = predictions.select( "label", "prediction")

rFcFcountTotal = predictions.count()
print("Total Count          = %g"%rFcFcountTotal)

rFcFcorrect=lp.filter(col('label')== col('prediction')).count()
print("Total Correct        = %g"%rFcFcorrect)
rFcFwrong = lp.filter(~(col('label') == col('prediction'))).count()
print("Total Wrong          = %g"%rFcFwrong)

rFcFratioWrong=float(float(rFcFwrong)/float(rFcFcountTotal))
print("Wrong Ratio          = %g"%rFratioWrong)
rFcFratioCorrect=float(float(rFcFcorrect)/float(rFcFcountTotal))
print("Right Ratio          = %g"%rFratioCorrect)
print("\n*********************************************************\n")

with open('wisdm_main_ver_0.0/main_result/additional_param.csv', mode='a') as paramFile:
    fieldnames = ['Classifier','Count Total','Correct','Wrong','Ratio Wrong','Ratio Correct','F1 Score','Training Time','Testing Time','Accuracy']
    writer = csv.DictWriter(paramFile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Classifier':lrModel,'Count Total':lRcountTotal,'Correct':lRcorrect,'Wrong':lRwrong,\
                     'Ratio Wrong':lRratioWrong,'Ratio Correct':lRratioCorrect,'F1 Score':lRf1,\
                     'Training Time':lRtt,'Testing Time':lRst,'Accuracy':lRaccuracy})
    writer.writerow({'Classifier':dtModel,'Count Total':dTcountTotal,'Correct':dTcorrect,'Wrong':dTwrong,\
                     'Ratio Wrong':dTratioWrong,'Ratio Correct':dTratioCorrect,'F1 Score':dTf1,\
                     'Training Time':lRtt,'Testing Time':lRst,'Accuracy':dTaccuracy})
    writer.writerow({'Classifier':rfModel,'Count Total':rFcountTotal,'Correct':rFcorrect,'Wrong':rFwrong,\
                     'Ratio Wrong':rFratioWrong,'Ratio Correct':rFratioCorrect,'F1 Score':rFf1,\
                     'Training Time':lRtt,'Testing Time':lRst,'Accuracy':rFaccuracy})

with open('wisdm_main_ver_0.0/main_result/crossFold_additional_param.csv', mode='a') as cVFile:
    fieldnames = ['Classifier','Count Total','Correct','Wrong','Ratio Wrong','Ratio Correct','F1 Score','Cross Validation Training Time','Cross Validation Testing Time','Cross Fold Accuracy']
    writer = csv.DictWriter(cVFile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Classifier':lrModel,'Count Total':lRcFcountTotal,'Correct':lRcFcorrect,'Wrong':lRcFwrong,\
                     'Ratio Wrong':lRcFratioWrong,'Ratio Correct':lRcFratioCorrect,'F1 Score':lRcFf1,\
                     'Cross Validation Training Time':lRcFtt,'Cross Validation Testing Time':lRcFst,'Cross Fold Accuracy':lRcFaccuracy})
    writer.writerow({'Classifier':dtModel,'Count Total':dTcFcountTotal,'Correct':dTcFcorrect,'Wrong':dTcFwrong,\
                     'Ratio Wrong':dTcFratioWrong,'Ratio Correct':dTcFratioCorrect,'F1 Score':dTcFf1,\
                     'Cross Validation Training Time':dTcFtt,'Cross Validation Testing Time':dTcFst,'Cross Fold Accuracy':dTcFaccuracy})
    writer.writerow({'Classifier':rfModel,'Count Total':rFcFcountTotal,'Correct':rFcFcorrect,'Wrong':rFcFwrong,\
                     'Ratio Wrong':rFcFratioWrong,'Ratio Correct':rFcFratioCorrect,'F1 Score':rFcFf1,\
                     'Cross Validation Training Time':rFcFtt,'Cross Validation Testing Time':rFcFst,'Cross Fold Accuracy':rFcFaccuracy})
    
res.close()                                                            #Comment to run in Jupyter

#=========================MATPLOT=====================================
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('seaborn-deep')

# Plots--------------------------------------------------------------")
numeric_data = data.select(numeric_features).sample(False, 0.10).toPandas()
n = len(numeric_data.columns)
for i in range(n):
    for j in range(n):
        ax = plt.subplot(numeric_data.plot.hexbin(x=i,y=j,sharex=False))
        xlbl = ax.xaxis.get_label_text()
        ylbl = ax.yaxis.get_label_text()
        plt.savefig(r"wisdm_main_ver_0.0/plot/Fig: "+str(xlbl)+"_"+str(ylbl)+".png")

axs = pd.scatter_matrix(numeric_data,diagonal='hist', alpha=0.2, figsize=(16,16));
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
plt.savefig(r"wisdm_main_ver_0.0/plot/Scatter_Matrix.png")