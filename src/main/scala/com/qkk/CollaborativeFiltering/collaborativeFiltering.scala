package com.qkk.CollaborativeFiltering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Collaborative fitering exercise
 * @author； qkk
 * @date: 20190914
 */
object collaborativeFiltering {
  def main(args: Array[String]): Unit = {
    // create spark object
    var conf  = new SparkConf().setAppName("collaborative filtering").setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load data
    var dataPath = "src/main/resources/sample_itemcf2.txt"
    var data = sc.textFile(dataPath)
    var userData = data.map(_.split(",")).map(line => ItemPref(line(0), line(1), line(2).toDouble)).cache()

    // compute similarity
    val simi = new ItemSimilarity()
    var simi_rdd = simi.Similarity(userData, "cooccurrence")

    // recommend
    var recomd = new RecommendedItem()
    var recomd_rdd = recomd.Recommend(simi_rdd, userData, 30)

    // print similarity matrix
    println(s"similarity matrix between items: ${recomd_rdd.count()} ")
    simi_rdd.collect().foreach{
      ItemSimi => println(ItemSimi.itemId1 + ", " + ItemSimi.itemId2 + ": " + ItemSimi.similarity)
    }

    // print user recommend list
    println(s"user recommend list: ${recomd_rdd.count()}")
    recomd_rdd.collect().foreach{
      UserRecomm => println(UserRecomm.userId + ", " + UserRecomm.itemId + ", " + UserRecomm.pref)
    }

  }
}
