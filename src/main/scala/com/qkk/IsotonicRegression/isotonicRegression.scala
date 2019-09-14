package com.qkk.IsotonicRegression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.{IsotonicRegression, IsotonicRegressionModel}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * isotonic regression exercise
 *
 * @author: qkk
 * @date: 20190914
 */
object isotonicRegression {
  def main(args: Array[String]): Unit = {
    // create Spark object
    var conf = new SparkConf().setAppName("isotonic regression").setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load data
    var dataPath = "src/main/resources/sample_isotonic_regression_data.txt"
    var data = sc.textFile(dataPath)
    var parseData = data.map{ line =>
      var parts = line.split(",").map(_.toDouble)
      (parts(0), parts(1), 1.0) /* label-feature-weight */
    }

    // split data
    var splits = parseData.randomSplit(Array(0.6, 0.4), seed = 11L)
    var (trainData, testData) = (splits(0), splits(1))

    // train
    var model = new IsotonicRegression()
      .setIsotonic(true)
      .run(trainData)

    //
    var x = model.boundaries
    var y = model.predictions
    println("boundaries \t predictions" )
    for (i <- 0 to x.length-1) {
      println(x(i) + "\t" + y(i))
    }

    // assembly prediction
    val predictionAndLabel = testData.map{
      line =>
        var prediction = model.predict(line._2)
        (prediction, line._1)
    }

    // print prediction
    var print_predict = predictionAndLabel.take(50)
    println("prediction \t label")
    for (i <- 0 to print_predict.length-1) {
      println(print_predict(i)._1 + ", " + print_predict(i)._2)
    }

    // print RMSE
    var rmse = predictionAndLabel.map{
      case (p, l) =>
        math.pow((p - l), 2)
    }.mean()
    println("Mean squared error: " + rmse)

    // save and load model
    val modelPath = "src/main/scala/com/qkk/IsotonicRegression/IsotonicRegressionModelRes"
    model.save(sc, modelPath)
    var lmodel  = IsotonicRegressionModel.load(sc, modelPath)
  }
}
