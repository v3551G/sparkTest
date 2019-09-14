package com.qkk.DecisionTree

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.model.DecisionTreeModel._
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Decision tree for regression exercise
 *
 * @author: qkk
 * @date: 20190913
 */
object decisionTreeForRegression {
  def main(args: Array[String]): Unit = {
    // create spark object
    var conf = new SparkConf().setAppName("decision for regression").setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load Data
    var data_path = "src/main/resources/lpsa.data"
    var data = sc.textFile(data_path)
    var examples = data.map{
      line =>
        var parts = line.split(",")
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }.cache()
    var numExamples = examples.count()

    // split data
    var splits = examples.randomSplit(Array(0.7, 0.3), seed = 21)
    var (trainData, testData) = (splits(0), splits(1))

    //  train
    var categoricalFeaturesInfo = Map[Int, Int]()
    var impurity = "variance"
    var maxDepth = 5
    var maxBins  = 32
    var model = DecisionTree.trainRegressor(trainData, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // print trained tree structure
    println("Learned tree structure:")
    println(model.toDebugString)

    // assembly prediction
    var labelAndPrediction = testData.map{
      line =>
        var prediction = model.predict(line.features)
        (line.label, prediction)
    }

    // print rmse
    var rmse = labelAndPrediction.map{
      case (l, e) =>
        var err = l-e
        err * err
    }.reduce(_ + _)
    println("rmse = " + rmse)

    // print prediction
    var print_predict = labelAndPrediction.take(50)
    println("label \t\t prediction")
    for (i <- 0 to print_predict.length-1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    // save and load model
    var modelPath = "src/main/scala/com/qkk/DecisionTree/DecisionTreeForRegressionModelRes";
    model.save(sc, modelPath)
    var lmodel = DecisionTreeModel.load(sc, modelPath)

  }
}
