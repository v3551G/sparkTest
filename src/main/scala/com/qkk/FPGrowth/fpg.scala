package com.qkk.FPGrowth

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.{SparkConf, SparkContext}

/**
 * FPGrowth exercise
 * @author: qkk
 * @date: 20190913
 */
object fpg {
  def main(args: Array[String]): Unit = {
    // create Spark object
    var conf = new SparkConf().setAppName("lda").setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load data
    var dataPath = "src/main/resources/sample_fpgrowth.txt";
    var data = sc.textFile(dataPath)
    var parseData = data.map(line => line.split(" ")).cache()

    // train
    var minSupport = 0.2
    var numPartation = 10
    var model = new FPGrowth()
        .setMinSupport(minSupport)
        .setNumPartitions(numPartation)
        .run(parseData)

    // print frequent items
    var freqItems = model.freqItemsets
    println(s"Number of frequent item sets: ${freqItems.count()}")
    freqItems.collect().foreach{
      item =>
        println(item.items.mkString("[", ",", "]") + ", " + item.freq)
    }
  }
}
