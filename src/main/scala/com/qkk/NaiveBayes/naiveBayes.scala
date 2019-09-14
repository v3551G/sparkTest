package com.qkk.NaiveBayes

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Naive Bayes exercise
 *@author: qkk
 *@date: 20190913
 */
object naiveBayes {
  def main(args: Array[String]): Unit = {
    // create spark object
    var conf = new SparkConf().setAppName("Naive Bayes").setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load data
    var dataPath = "src/main/resources/sample_naive_bayes_data.txt";
    var data = sc.textFile(dataPath)
    var parseData = data.map{
      line =>
        var parts = line.split(",")
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }

    // split data
    var splits = parseData.randomSplit(Array(0.6, 0.4), seed = 132)
    var trainData = splits(0)
    var testData = splits(1)

    // train
    var model = NaiveBayes.train(trainData, lambda = 1.0, modelType = "multinomial")

    // assembly result
    var predictionAndLabel = testData.map{
      line =>
        var prediction = model.predict(line.features)
        (prediction, line.label)
    }

    // print
    var print_predict = predictionAndLabel.take(50)
    println("prediction  \t\t label")
    for (i <- 0 to print_predict.length -1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    // accuracy
    var accuracy = 1.0 * predictionAndLabel.filter(line => line._1 == line._2).count() / testData.count()
    print("accuracy: " + accuracy)

    // save and load model
    var modelPath = "src/main/scala/com/qkk/NaiveBayes/NaiveBayesModelRes"
    model.save(sc, modelPath)
    var lmodel = NaiveBayesModel.load(sc, modelPath)

  }
}
