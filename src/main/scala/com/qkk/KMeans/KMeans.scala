package com.qkk.KMeans

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
 * KMeans cluster exercise
 * @author: qkk
 * @date: 20190913
 */
object KMeans {
  def main(args: Array[String]): Unit = {
    // new Spark object
    var conf = new SparkConf()
      .setAppName("KMeans")
      .setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load Data
    var dataPath = "src/main/resources/kmeans_data.txt"
    var data = sc.textFile(dataPath)
    var parseData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // new KMeans model and training
    var initMode = "k-means"
    var numClusters = 2
    var numIterators = 20
    var model = new KMeans()
      .setInitializationMode(initMode)
      .setK(numClusters)
      .setMaxIterations(numIterators)
      .run(parseData)

    // compute error
    var wsse = model.computeCost(parseData)
    println("Sum of squared errors: " + wsse)
    // print cluster center
    var clusterCenters = model.clusterCenters
    println("cluster center: \n")
    for (item <- clusterCenters) {
      println(item.toString)
    }
    // test belongs of new instance
    var testData = "0.11 0.3 0.3".split(" ").map(_.toDouble)
    var belong = model.predict(Vectors.dense(testData))
    println(testData.toString + " belongs to cluster: " + belong)

    // assembly result
    println("clustering result:")
    val result = data.map{
       line =>
        var lineVector = Vectors.dense(line.split(" ").map(_.toDouble))
        var prediction = model.predict(lineVector)
        lineVector + " "  + prediction
    }.collect.foreach(println)

    // save and load model
    var modelPath = "src/main/scala/com/qkk/KMeans/KMeansModelRes"
    model.save(sc, modelPath)
    var lmodel = KMeansModel.load(sc, modelPath)

    sc.stop
  }
}
