from BoxCoxTransformer import BoxCoxTransformer
from pyspark import SQLContext
from pyspark import SparkContext, Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Bucketizer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np

if __name__ == "__main__":
    # read data
    SAVE_MODELS = False
    sc = SparkContext(appName="Cancer Analysis")
    raw_data = sc.textFile("./data/cancer_data.txt")
    sqlContext = SQLContext(sc)
    raw_parts = raw_data.map(lambda x: x.split(',', -1))
    # labels: Cancer, Gender, Age, Weight, Height, Job
    parts = raw_parts.map(lambda x: Row(label=(x[0]), Gender=x[1], Age=(x[2]),
                                        Weight=(x[3]), Height=(x[4]), Job=x[5]))

    schemaData = sqlContext.createDataFrame(parts)
    schemaData.registerTempTable('CancerData')
    # printing info
    # schemaData.printSchema()

    # converting data
    df = schemaData.withColumn('labelTmp', schemaData.label.cast('double')).drop('label').withColumnRenamed('labelTmp',
                                                                                                            'label') \
        .withColumn('WeightTmp', schemaData.Weight.cast('double')).drop('Weight').withColumnRenamed('WeightTmp',
                                                                                                    'Weight') \
        .withColumn('HeightTmp', schemaData.Height.cast('double')).drop('Height').withColumnRenamed('HeightTmp',
                                                                                                    'Height') \
        .withColumn('AgeTmp', schemaData.Age.cast('double')).drop('Age').withColumnRenamed('AgeTmp', 'Age')

    # df.printSchema()


    # indexers
    jobIndexer = StringIndexer(inputCol='Job', outputCol='JobIndex')
    genderIndexer = StringIndexer(inputCol='Gender', outputCol='GenderIndex')

    # age splits
    ageSplits = [0.0, 5.0, 12.0, 18.0, 30.0, 60.0, 120.0, float('inf')]
    ageBucketizer = Bucketizer(splits=ageSplits, inputCol="Age", outputCol="bucketedAge")

    # OneHotEncoder
    jobEncoder = OneHotEncoder(dropLast=False, inputCol="JobIndex", outputCol="JobVec")
    genderEncoder = OneHotEncoder(dropLast=False, inputCol="GenderIndex", outputCol="GenderVec")

    # boxcox transformer
    ageBoxCox = BoxCoxTransformer(inputCol='Age', outputCol='AgeT', alpha=0.54442)
    weightBoxCox = BoxCoxTransformer(inputCol='Weight', outputCol='WeightT', alpha=0.15431)
    heightBoxCox = BoxCoxTransformer(inputCol='Height', outputCol='HeightT', alpha=0.9695)
    # debug
    # ageBoxCox.transform(df).show()

    featureAssembler = VectorAssembler(inputCols=["JobVec", "GenderVec", "AgeT", "WeightT", "HeightT"],
                                       outputCol="features")

    currPipeline = [jobIndexer, genderIndexer, jobEncoder, genderEncoder, ageBoxCox,ageBucketizer, weightBoxCox, heightBoxCox,
                    featureAssembler]

    # trainer
    NUMBER_OF_MODELS = 10
    dataPipeline = Pipeline(stages=currPipeline)
    convertedData = dataPipeline.fit(df).transform(df)
    trainData, testData = convertedData.randomSplit([0.9, 0.1], seed=0)
    models = []

    for i in range(NUMBER_OF_MODELS):
        tmpSample = trainData.sample(True, 1.0, 42)
        reg = LogisticRegression(labelCol="label", featuresCol="features", standardization=False)
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
        cvPipeLine = Pipeline(stages=[reg])
        paramGrid = ParamGridBuilder().addGrid(reg.regParam, [0.01, 0.1, 1, 2, 5]).build()
        crossval = CrossValidator(
            estimator=cvPipeLine,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=2)

        tmpModel = crossval.fit(tmpSample)
        models.append(tmpModel)


    # save models
    if SAVE_MODELS:
        modelIndex = 0
        for elem in models:
            elem.save("./savedModels/model_" + str(modelIndex))
            modelIndex = modelIndex + 1

    trainSize = trainData.count()
    allResults = np.zeros(trainSize)

    # testing some results
    for model in models:
        tResult = model.transform(trainData)
        preds = tResult.select(['prediction']).collect()
        preds = np.array([float(p.prediction) for p in preds])
        np.add(allResults, preds)

    preds = np.divide(preds, trainSize)
    zeroIndex = preds < 0.5
    preds[zeroIndex] = 0.0
    preds[preds > 0.0] = 1.0
    actualLabes = trainData.select(['label']).collect()
    actualLabes = np.array([float(a.label) for a in actualLabes])
    errCount = 0.0
    for elem in zip(preds, actualLabes):
        if elem[0] != elem[1]:
            errCount = errCount + 1.0

    print("ACC: " + str(1 - errCount / float(trainSize)))
