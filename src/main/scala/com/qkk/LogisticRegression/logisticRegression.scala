package com.qkk.LogisticRegression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Linear Regression exercise
 * @author: qkk
 * @date: 20190913
 */
object logisticRegression {
  def main(args: Array[String]): Unit = {
    // create Spark object
    var conf = new SparkConf()
      .setAppName("logistic regression")
      .setMaster("local")
    var sc = new  SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load data
    var dataPath = "src/main/resources/sample_libsvm_data.txt";
    var data = MLUtils.loadLibSVMFile(sc, dataPath)

    // split data
    var splitRate = Array(0.6, 0.4)
    var splits= data.randomSplit(splitRate, seed = 123)
    var trainData = splits(0)
    var testData = splits(1)

    // train model
    var model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(trainData)
    /*
    var model2  = new LogisticRegressionWithSGD()
      .run(trainData)
    */

    // assembly prediction
    var predictionAndLabels = testData.map{
      case LabeledPoint(label, features) =>
        var prediction = model.predict(features)
        (prediction, label)
    }

    // print
    var print_label = predictionAndLabels.take(50)
    println("prediction  \t\t label")
    for (i <- 0 to print_label.length-1) {
      println(print_label(i)._1 + " \t" + print_label(i)._2)
    }

    // evaluate
    var metrics = new MulticlassMetrics(predictionAndLabels)
    var precision = metrics.precision
    println("precision: " + precision)
    var accuracy = metrics.accuracy
    println("accuracy: " + accuracy)
    var fMeasure = metrics.fMeasure
    println("fMeasure: " + fMeasure)

    // save and load model
    var modelPath = "src/main/scala/com/qkk/LogisticRegression/LogisticRegressionModelRes"
    model.save(sc, modelPath)
    var lmodel = LogisticRegressionModel.load(sc, modelPath)

  }
}
