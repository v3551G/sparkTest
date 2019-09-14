package com.qkk.LinearRegression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

/**
 *Liner Reegression exercise
 * @author: qkk
 * @date: 20190913
 */
object LinearRegression {
  def main(args: Array[String]): Unit = {
    // create Spark object
    var conf = new SparkConf().setAppName("LR").setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load data
    var data_path = "src/main/resources/lpsa.data"
    var data = sc.textFile(data_path)
    var examples = data.map{
      line =>
        var parts = line.split(",")
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }.cache()
    var numExamples = examples.count()

    // train
    var numIterations = 100
    var stepSize = 1
    var miniBatchFraction = 1
    var model = LinearRegressionWithSGD.train(examples, numIterations, stepSize, miniBatchFraction)

    // print parameter
    println("model weights: " + model.weights)
    println("model intercept: " + model.intercept)

    // assembly prediction result
    var prediction = model.predict(examples.map(_.features))
    var predictionAndLabel = prediction.zip(examples.map(_.label))
    var print_predict = predictionAndLabel.take(50);
    println("prediction \t\t  label")
    for (i <- 0 to print_predict.length -1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    // compute rmse
    var loss = predictionAndLabel.map{
      case (p, l) =>
        var err = p-l
        err * err
    }.reduce(_ + _)
    var rmse = math.sqrt(loss / numExamples)
    println(s"Test RMSE = $rmse.")
    println(s"Test RMSE = ${rmse}.")

    // save and load model
    var modelPath = "src/main/scala/com/qkk/LinearRegression/LinerRegressionModelRes"
    model.save(sc, modelPath)
    var lmodel = LinearRegressionModel.load(sc, modelPath)

  }
}
