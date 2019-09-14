package com.qkk.DecisionTree

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Decision tree for classification exercise
 *@author: qkk
 *@date: 20190913
 */
object decisionTree {
   def main(args: Array[String]): Unit = {
     // create Spark object
     var conf = new SparkConf().setAppName("decisionTree").setMaster("local")
     var sc = new SparkContext(conf)
     Logger.getRootLogger.setLevel(Level.WARN)

     // load data
     var dataPath = "src/main/resources/sample_libsvm_data.txt"
     var data = MLUtils.loadLibSVMFile(sc, dataPath)

     // split data
     var splits = data.randomSplit(Array(0.7, 0.3), seed = 11)
     var (trainData, testData) = (splits(0), splits(1))

     // train
     var numClasses = 2
     var categoricalFeaturesInfo = Map[Int, Int]()
     var impurity = "gini"
     var maxDepth = 5
     var maxBins = 32
     var model = DecisionTree.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

     // assembly prediction
     var labelAndPreds = testData.map{ instance =>
       var prediction = model.predict(instance.features)
       (instance.label, prediction)
     }

     // print
     var print_predict  = labelAndPreds.take(20)
     println("label \t\t prediction")
     for (i <- 0 to print_predict.length-1) {
       println(print_predict(i)._1 + "\t\t" + print_predict(i)._2)
     }

     // compute error rate
     var testErr = labelAndPreds.filter(line => line._1 != line._2).count().toDouble / testData.count()
     println("test error: " + testErr)
     var modelString = model.toDebugString
     println("Learned classification tree model: \n" + modelString)

     // save and load model
     var modelPath = "src/main/scala/com/qkk/DecisionTree/DecisionTreeModelRes";
     model.save(sc, modelPath)
     var lmodel = DecisionTreeModel.load(sc, modelPath)
   }
}
