package com.qkk.LDA

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
 * lda exercise
 * @author: qkk
 * @date: 20190913
 */
object lda {
  def main(args: Array[String]): Unit = {
    // create Spark object
    var conf = new SparkConf().setAppName("lda").setMaster("local")
    var sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // load data
    var dataPath = "src/main/resources/sample_lda_data.txt";
    var data = sc.textFile(dataPath)
    var parseData = data.map(line => Vectors.dense(line.split(" ").map(_.toDouble)))

    var pDi = parseData.zipWithIndex.map(_.swap).cache()

    // train
    var topicNumber = 3
    var docHyperParameter_alpha = 5
    var topicHyperParameter_beta = 5
    var seed = 10
    var checkPointInterval = 10
    var ldaOptimizer = "em" /* em | online*/
    var model = new LDA()
      .setK(topicNumber)
      .setDocConcentration(docHyperParameter_alpha)
      .setTopicConcentration(topicHyperParameter_beta)
      .setSeed(seed)
      .setCheckpointInterval(checkPointInterval)
      .setOptimizer(ldaOptimizer)
      .run(pDi)

    // print topic distribution
    println("Learned topics  distribution (as distributions over vocab of " + model.vocabSize + " words):")
    var topics = model.topicsMatrix /* vocab * topic */
    for (topic <- Range(0, 3)) {
      println("Topic " + topic + " :")
      for (vocab <- Range(0, model.vocabSize)) {
        print(" " + topics(vocab, topic))
      }
      println()
    }

    // print sorted topic distribution
    var sortedTopics =  model.describeTopics(4)
    println("Sorted topic distributions: ")
    for (topic <- 0 to sortedTopics.length-1) {
      var (vocabs, weights) = (sortedTopics(topic)._1, sortedTopics(topic)._2)
      var vocStr = ""
      var weiStr = ""
      for (vocab <-0 to vocabs.length-1) {
        vocStr += vocabs(vocab) + ", "
        weiStr += weights(vocab) + ", "
      }
      println(vocStr + "-----" + weiStr)
    }

    // print topic distribution of doc
    var distLDAModel = model.asInstanceOf[DistributedLDAModel]
    var docDistributions = distLDAModel.topicDistributions.collect
    println("doc distribution of topics: \n")
    for (doc <- 0 to docDistributions.length-1) {
      var id = docDistributions(doc)._1
      var distribution = docDistributions(doc)._2
      println("id: " + id + " : " + distribution)
    }
  }
}
