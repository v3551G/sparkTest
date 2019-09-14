package com.qkk.SVM

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
 * SVM exercise
 * @author: qkk
 * @date： 20190913
 */
object libSvm {
  def main(args: Array[String]): Unit = {
    // create spark object
    var conf = new SparkConf().setAppName("svm").setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load data
    var dataPath = "src/main/resources/sample_libsvm_data.txt";
    var data = MLUtils.loadLibSVMFile(sc, dataPath)

    // split data
    var splits = data.randomSplit(Array(0.6, 0.4), seed = 11)
    var trainData = splits(0)
    var testData = splits(1)

    // train
    var numIterations = 100
    var model = SVMWithSGD.train(trainData, numIterations)

    // assembly prediction
    var predictionAndLabel = testData.map{
      point =>
        var score = model.predict(point.features)
        (score, point.label)
    }

    // print
    var print_predict = predictionAndLabel.take(10)
    println("prediction \t\t label")
    for (i <- 0 to print_predict.length-1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    // accuray
    var accuracy = 1.0 * predictionAndLabel.filter(line => line._1 == line._2).count() / testData.count()
    println("Area under ROC = " + accuracy)

    // save and load model
    var modelPath = "src/main/scala/com/qkk/SVM/SVMModelRes"
    model.save(sc, modelPath)
    var lmodel = SVMModel.load(sc, modelPath)
  }
}
