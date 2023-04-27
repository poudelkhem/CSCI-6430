

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

install_and_import('findspark')
install_and_import('pyspark')
# install_and_import('imbalanced-learn')

# Import statements
findspark.init()
findspark.find()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create spark session and read in data
spark = SparkSession.builder.config('spark.ui.showConsoleProgress', 'false').getOrCreate()
data = spark.read.csv('data.csv', header=True, inferSchema=True)

# Handle Imbalanced Classes
import numpy as np
from imblearn.over_sampling import SMOTE

# create the  object with the desired sampling strategy.
smote = SMOTE()
dfp = data.toPandas()

# fit the object to our training data
data2, y = smote.fit_resample(dfp.loc[:,dfp.columns!='diagnosis'], dfp['diagnosis'])

_, class_counts = np.unique(y, return_counts=True)
class_names = ['Malignant', 'Benign']
data2["diagnosis"] = y

# Shuffle the data for Monty Carlo sampling
data2 = shuffle(data2)

# Turn the data frame back into a Spark data frame
df = spark.createDataFrame(data2)

# Drop unnecessary column
df = df.drop("id")

# Create label and feature encoder
Label_encoder = StringIndexer(inputCol="diagnosis",outputCol="diagnosis"+"Label").fit(df)
feature_cols = df.columns[0:-1]
featureAssembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Prepare the data
dataNew = featureAssembler.transform(Label_encoder.transform(df))
(trainingData, testData) = dataNew.randomSplit([0.7, 0.3])

# Create empty for list
models = []

# Decision tree model construction
from pyspark.ml.classification import DecisionTreeClassifier

# Define the DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="diagnosisLabel", featuresCol="features", maxDepth=25)

# Hyperparameter tuning
paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15, 20, 25]).build()

crossval = CrossValidator(estimator=dt,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="diagnosisLabel"),
                          parallelism=2,
                          numFolds=5)  # use 3+ folds in practice
# Train the DecisionTreeClassifier
decision_tree = crossval.fit(trainingData)
models.append(("Decision Tree", decision_tree))

# Random forest model construction
from pyspark.ml.classification import RandomForestClassifier

# Define the random forest model construction
rf = RandomForestClassifier(labelCol="diagnosisLabel", featuresCol="features", numTrees=100, maxDepth=25)

# Hyperparameter tuning and cross validation
paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10, 15, 20, 25]) \
            .addGrid(rf.numTrees, [10, 25, 50, 75, 100, 150, 200]).build()

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="diagnosisLabel"),
                          parallelism=2,
                          numFolds=5)  # use 3+ folds in practice

# Train the random forest model construction
random_forest = rf.fit(trainingData)
models.append(("Random Forest", random_forest))

# Multi layer perceptron classifier model construction
from pyspark.ml.classification import MultilayerPerceptronClassifier

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [30, 256, 128, 2]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(labelCol="diagnosisLabel", featuresCol="features", maxIter=100, layers=layers, blockSize=128, seed=1234)

paramGrid = ParamGridBuilder().addGrid(trainer.maxIter, [25, 50, 100, 200]).build()

crossval = CrossValidator(estimator=trainer,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="diagnosisLabel"),
                          parallelism=2,
                          numFolds=5)  # use 3+ folds in practice

# train the model
ML_perceptron = crossval.fit(trainingData)
models.append(("ML perceptron", ML_perceptron))

# Predict the model stuff
for item in models:
    model = item[1]
    # Make predictions
    predictions = model.transform(testData)

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(labelCol="diagnosisLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(str(item[0]), ":" , accuracy)
