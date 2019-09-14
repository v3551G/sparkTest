package com.qkk.CollaborativeFiltering

import org.apache.spark.rdd.RDD
import scala.math._
/**
 * user comment score
 *
 * @param userId
 * @param itemId
 * @param pref
 */
case class ItemPref(val userId: String,
                   val itemId: String,
                   val pref: Double
                   ) extends Serializable

/**
 * user recommendation
 * @param userId
 * @param itemId
 * @param pref
 */
case class UserRecomm(
                     val userId: String,
                     val itemId: String,
                     val pref: Double
                     ) extends Serializable

/**
 * Similarity between items
 * @param itemId1
 * @param itemId2
 * @param similarity
 */
case class ItemSimi(
                   val itemId1: String,
                   val itemId2: String,
                   val similarity: Double
                   ) extends Serializable

/**
 * Similarity class defination
 */
class ItemSimilarity extends Serializable {
  /**
   * compute similarity
   * @param user_rdd
   * @param sType: similarity type
   * @return
   */
  def Similarity(user_rdd: RDD[ItemPref], sType: String): (RDD[ItemSimi]) = {
    var similarity_rdd = sType match{
      case "cooccurrence" =>
        ItemSimilarity.CooccurrenceSimilarity(user_rdd)
      case "cosine" =>
        ItemSimilarity.CosineSimlarity(user_rdd)
      case "euclidean" =>
        ItemSimilarity.EuclideanSimilarity(user_rdd)
    }
    similarity_rdd
  }
}


/**
 * ItemSimilarity object
 */
object ItemSimilarity {
  /**
   *
   * @param user_rdd
   * @return
   */
  def CooccurrenceSimilarity(user_rdd: RDD[ItemPref]): (RDD[ItemSimi]) = {

    var user_rdd1 = user_rdd.map(line => (line.userId, line.itemId, line.pref))
    var user_item = user_rdd1.map(line => (line._1, line._2)) /* (i,j)*/

    var u_iJoin = user_item.join(user_item) /*(i,(j,k))*/
    var iic = u_iJoin.map(line => (line._2, 1)) /* ((j,k),1) */

    var iiCount = iic.reduceByKey((c1,c2) => c1 + c2) /* ((j,k),c) */

    var symPart = iiCount.filter(line => line._1._1 == line._1._2)  /* ((i,i),c)*/
    var asymPart = iiCount.filter(line => line._1._1 != line._1._2) /* ((i,j),c)*/

    var r1 = asymPart.map(r => (r._1._1, (r._1._1, r._1._2, r._2)))
      .join(symPart.map(r => (r._1._1, r._2))) /* (i,((i,j,c),c)) */
    var r2 = r1.map( r => (r._2._1._2, (r._2._1._1, r._2._1._2, r._2._1._3, r._2._2))) /* (j,(i,j,c,c)) */

    var r3 = r2.join(symPart.map(r => (r._1._1, r._2))) /* (j, ((i,j,c,c),c)) */
    var r4 = r3.map(r => (r._2._1._1, r._2._1._2, r._2._1._3, r._2._1._4, r._2._2)) /*(i,j,c,c,c)*/
    // compute similarity
    var iis = r4.map(r => (r._1, r._2, (r._3/sqrt(r._4 * r._5)))) /* (i,j,c) */

    // wrap  data
    iis.map(r => ItemSimi(r._1, r._2, r._3))

  }

  /**
   * Cosine similarity
   * @param user_rdd
   * @return
   */
  def CosineSimlarity(user_rdd: RDD[ItemPref]): (RDD[ItemSimi]) = {

    val user_rdd1 = user_rdd.map(f => (f.userId, f.itemId, f.pref))
    val user_rdd2 = user_rdd1.map(f => (f._1, (f._2, f._3)))

    val user_rdd3 = user_rdd2.join(user_rdd2)
    val user_rdd4 = user_rdd3.map(f => ((f._2._1._1, f._2._2._1), (f._2._1._2, f._2._2._2)))

    val user_rdd5 = user_rdd4.map(f => (f._1, f._2._1 * f._2._2)).reduceByKey(_ + _)

    val user_rdd6 = user_rdd5.filter(f => f._1._1 == f._1._2)

    val user_rdd7 = user_rdd5.filter(f => f._1._1 != f._1._2)

    val user_rdd8 = user_rdd7.map(f => (f._1._1, (f._1._1, f._1._2, f._2)))
      .join(user_rdd6.map(f => (f._1._1, f._2)))
    val user_rdd9 = user_rdd8.map(f => (f._2._1._2, (f._2._1._1,
      f._2._1._2, f._2._1._3, f._2._2)))
    val user_rdd10 = user_rdd9.join(user_rdd6.map(f => (f._1._1, f._2)))
    val user_rdd11 = user_rdd10.map(f => (f._2._1._1, f._2._1._2, f._2._1._3, f._2._1._4, f._2._2))
    val user_rdd12 = user_rdd11.map(f => (f._1, f._2, (f._3 / sqrt(f._4 * f._5))))

    user_rdd12.map(f => ItemSimi(f._1, f._2, f._3))
  }

  /**
   *
   * @param user_rdd
   * @return
   */
  def EuclideanSimilarity(user_rdd: RDD[ItemPref]): (RDD[ItemSimi]) = {
    val user_rdd1 = user_rdd.map(f => (f.userId, f.itemId, f.pref))
    val user_rdd2 = user_rdd1.map(f => (f._1, (f._2, f._3)))

    val user_rdd3 = user_rdd2 join user_rdd2
    val user_rdd4 = user_rdd3.map(f => ((f._2._1._1, f._2._2._1), (f._2._1._2, f._2._2._2)))

    val user_rdd5 = user_rdd4.map(f => (f._1, (f._2._1 - f._2._2) * (f._2._1 - f._2._2))).reduceByKey(_ + _)

    val user_rdd6 = user_rdd4.map(f => (f._1, 1)).reduceByKey(_ + _)

    val user_rdd7 = user_rdd5.filter(f => f._1._1 != f._1._2)

    val user_rdd8 = user_rdd7.join(user_rdd6)
    val user_rdd9 = user_rdd8.map(f => (f._1._1, f._1._2, f._2._2 / (1 + sqrt(f._2._1))))

    user_rdd9.map(f => ItemSimi(f._1, f._2, f._3))
  }
}
