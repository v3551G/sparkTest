package com.qkk.ALS

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * ALS exercise
 * @author: qkk
 * @date: 201909013
 */
object als {
  def main(args: Array[String]): Unit = {
    // create Spark object
    var conf = new SparkConf().setAppName("als").setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load data
    var dataPath = "src/main/resources/test.data"
    var data = sc.textFile(dataPath)
    /*
    var ratings = data.map(_.split(',')) match {
      case Array(user, item, rate) =>
        Rating(user.toInt, item.toInt, rate.toDouble)
    }
    */
    var ratings = data.map(_.split(',')).map(line => Rating(line(0).toInt, line(1).toInt, line(2).toDouble))

    // train
    var rank = 10
    var numIterations = 10
    var lamda = 0.01
    var model = ALS.train(ratings, rank, numIterations, lamda)

    // assembly prediction
    var user_items = ratings.map {
      case Rating(user, item, rate) =>
        (user, item)
    }

    var predictions = model.predict(user_items).map{
      case Rating(user, item, rate) =>
        ((user, item), rate)
    }

    var ratesAndPredictions = ratings.map{
      case Rating(user, item, rate) =>
        ((user, item), rate)
    }.join(predictions) /* ((user, item), (rate1, rate2)) */

    // compute RMSE
    var rmse = ratesAndPredictions.map {
      case (((user, item), (r1, r2))) =>
        val err = r1 - r2
        err * err
    }.mean()
    println("Mean Square errr = " + rmse)

    //
    println("user \t item \t rate \t prediction")
    val print_prediction = ratesAndPredictions.take(20)
    for (i <- 0 to print_prediction.length-1) {
      println(print_prediction(i)._1._1 + "\t\t" + print_prediction(i)._1._2 + "\t\t"
            + print_prediction(i)._2._1 + "\t\t" + print_prediction(i)._2._2)
    }

    //
    var modelPath = "src/main/scala/com/qkk/ALS/alsModelRes"
    model.save(sc, modelPath)
    var lmodel = MatrixFactorizationModel.load(sc, modelPath)



  }
}
