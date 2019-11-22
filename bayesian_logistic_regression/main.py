import numpy as np
from class_BBVI import BBVI

if __name__ == "__main__":
    trainSource='./dataSet/a9a-train.txt'
    testSource='./dataSet/a9a-test.txt'
    featureDim=123
    maxIteration=200
    batchSize=100
    sampleSize=1000
    stepScale=0.1
    startPara=np.zeros(2*(featureDim+1))
    dataAccessing='CA'
    interval=10
    testSampleNum=200

    test=BBVI(
        trainSource=trainSource,
        testSource=testSource,
        featureDim=featureDim,
        maxIteration=maxIteration,
        batchSize=batchSize,
        sampleSize=sampleSize,
        stepScale=stepScale,
        startPara=startPara,
        dataAccessing=dataAccessing,
        interval=interval,
        testSampleNum=testSampleNum
    )
    test._BBVI_basic()
    a=1
    