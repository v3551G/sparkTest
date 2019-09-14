package com.qkk.CollaborativeFiltering

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

/**
 *
 */
class RecommendedItem {
  /**
   * 用户推荐计算.
   * @param items_similar 物品相似度
   * @param user_prefer 用户评分
   * @param r_number 推荐数量
   * @param RDD[UserRecomm] 返回用户推荐物品
   *
   */
  def Recommend(items_similar: RDD[ItemSimi],
                user_prefer: RDD[ItemPref],
                r_number: Int): (RDD[UserRecomm]) = {

    val rdd_app1_R1 = items_similar.map(f => (f.itemId1, f.itemId2, f.similarity))
    val user_prefer1 = user_prefer.map(f => (f.userId, f.itemId, f.pref))

    val rdd_app1_R2 = rdd_app1_R1.map(f => (f._1, (f._2, f._3))).
      join(user_prefer1.map(f => (f._2, (f._1, f._3))))

    val rdd_app1_R3 = rdd_app1_R2.map(f => ((f._2._2._1, f._2._1._1), f._2._2._2 * f._2._1._2))

    val rdd_app1_R4 = rdd_app1_R3.reduceByKey((x, y) => x + y)

    val rdd_app1_R5 = rdd_app1_R4.leftOuterJoin(user_prefer1.map(f => ((f._1, f._2), 1))).
      filter(f => f._2._2.isEmpty).map(f => (f._1._1, (f._1._2, f._2._1)))

    val rdd_app1_R6 = rdd_app1_R5.groupByKey()
    val rdd_app1_R7 = rdd_app1_R6.map(f => {
      val i2 = f._2.toBuffer
      val i2_2 = i2.sortBy(_._2)
      if (i2_2.length > r_number) i2_2.remove(0, (i2_2.length - r_number))
      (f._1, i2_2.toIterable)
    })
    val rdd_app1_R8 = rdd_app1_R7.flatMap(f => {
      val id2 = f._2
      for (w <- id2) yield (f._1, w._1, w._2)
    })
    rdd_app1_R8.map(f => UserRecomm(f._1, f._2, f._3))
  }

  /**
   * 用户推荐计算.
   * @param items_similar 物品相似度
   * @param user_prefer 用户评分
   * @param RDD[UserRecomm] 返回用户推荐物品
   *
   */
  def Recommend(items_similar: RDD[ItemSimi],
                user_prefer: RDD[ItemPref]): (RDD[UserRecomm]) = {

    val rdd_app1_R1 = items_similar.map(f => (f.itemId1, f.itemId2, f.similarity))
    val user_prefer1 = user_prefer.map(f => (f.userId, f.itemId, f.pref))

    val rdd_app1_R2 = rdd_app1_R1.map(f => (f._1, (f._2, f._3))).
      join(user_prefer1.map(f => (f._2, (f._1, f._3))))

    val rdd_app1_R3 = rdd_app1_R2.map(f => ((f._2._2._1, f._2._1._1), f._2._2._2 * f._2._1._2))

    val rdd_app1_R4 = rdd_app1_R3.reduceByKey((x, y) => x + y)

    val rdd_app1_R5 = rdd_app1_R4.leftOuterJoin(user_prefer1.map(f => ((f._1, f._2), 1))).
      filter(f => f._2._2.isEmpty).map(f => (f._1._1, (f._1._2, f._2._1)))

    val rdd_app1_R6 = rdd_app1_R5.map(f => (f._1, f._2._1, f._2._2)).
      sortBy(f => (f._1, f._3))
    rdd_app1_R6.map(f => UserRecomm(f._1, f._2, f._3))
  }
}
