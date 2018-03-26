import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.collection.mutable.ListBuffer


object LSH_Jaccard {

  //hash function

  def get_prime(list: List[Int]): List[Int] = list match {
    case Nil => Nil
    case x :: xs => List(x) ::: get_prime(xs.filter(_ % x != 0))
  }

  def prime(n: Int) = {
    val list = (2 to n).toList
    get_prime(list)
  }

  val prime_nums = prime(1000)

  var a_list = ListBuffer.empty[Int]
  var b_list = ListBuffer.empty[Int]
  var m_list = ListBuffer.empty[Int]

  val hash_num = 270

  for(_ <- 0 until hash_num){

    val a_index = scala.util.Random.nextInt(150)+1
    val m_index = scala.util.Random.nextInt(150)+1
    val b = scala.util.Random.nextInt(400)+1

    b_list+= b
    a_list += prime_nums(a_index)
    m_list += prime_nums(m_index)

  }



  def main(args: Array[String]): Unit = {
    val t0 = System.nanoTime()
    val conf = new SparkConf()
    conf.setAppName("LSH_Jaccard")
    conf.setMaster("local[*]")


    val sc = new SparkContext(conf)


    // csv:  userID, productID, rating
    val file_data = sc.textFile("data/video_small_num.csv").cache()
    var rdd = file_data.map(line => line.split(","))
    val header = rdd.first()
    rdd = rdd.filter(_ (0) != header(0))


    //list of distinct user
    val user = rdd.map(x => x(0).toInt).distinct()


    val maxIndex = user.max + 1

    //println(maxIndex)

    val userProductRatings = rdd.map(line => (line(1).toInt, (line(0).toInt, 1.toDouble)))
      .groupByKey().map(line => (line._1, line._2.toSeq)) //(productID,((userID1,1),(userID2,1),(userID3,1)...))

    val sparseVectorData = userProductRatings.map { line =>
      val sv: Vector = Vectors.sparse(maxIndex.toInt, line._2)

      (line._1, sv)
    }
   // println("!!!Sparse matrix!!!")
   // sparseVectorData.collect().foreach(println)



    //minhash

    val rdd_minhash = sparseVectorData.map { line =>

      var list_after_hash = ListBuffer.empty[Int]

      val user_index_list = line._2.toSparse.indices.toList


     val minhash_list = minhash(user_index_list)

      (line._1,(minhash_list,user_index_list))//(productID,(signature,sp_index))
    }



    val band_num = 90
    val row_num = hash_num/band_num



    val rdd_band = rdd_minhash.map{case(pId,(signature,sp_index)) =>
      var band_list = ListBuffer.empty[Tuple2[ListBuffer[Int],Int]]// (sub_list,band number)


      for(i <-0 until band_num){
        val sub_list = signature.slice(i*row_num,i*row_num+row_num)
        band_list += Tuple2(sub_list,i)
      }

      ((pId,sp_index),band_list)   //(productID,sp_index),band_list)
    }



   val band_divide = rdd_band.flatMap{ case ((pId,sp_index),band_list)  =>

     for(elem <- band_list) yield (elem,(pId,sp_index))//(band,(productID,sp_index))

    }
      .groupByKey()
     .map(line =>line._2.toList)
      .filter(line =>line.size>=2)



        val pair_rdd = band_divide.map { line =>

       var pair_list = ListBuffer.empty[Tuple2[(Int,List[Int]),(Int,List[Int])]]
          for(i <- 0 until line.size){
                  for( j<- i+1 until line.size){
                    pair_list += Tuple2(line(i),line(j))
                    }
                  }
          pair_list
        }

    val cdd_pair = pair_rdd.collect().flatten.toSet


    val result  = cdd_pair.map{case((p1,f1),(p2,f2))=>

      val union_num = f1.union(f2).distinct.length

      val intersect_num = f1.intersect(f2).length

      val j_sim = intersect_num.toDouble/union_num.toDouble


      (List(p1,p2).sorted,j_sim)
        }.filter(line=>line._2>=0.5)



    val output_file = new File("data/similarItem_jaccard.txt")
        val out = new PrintWriter(output_file)
        for (elem <- result){

        out.write(elem._1(0) + "," + elem._1(1) +","+ elem._2 +"\n")


        }

        out.close()


        // ground truth
        val answer_data = sc.textFile("Data/video_small_ground_truth_jaccard.csv").map(line => line.split(","))
          .map(line => Set(line(0).toInt, line(1).toInt))

        val my_result = sc.parallelize(result.toSeq)
          .map(line =>Set(line._1(0),line._1(1)))



        val tp = answer_data.intersection(my_result).count()


        val fn =answer_data.subtract(my_result).count()

        val fp = my_result.subtract(answer_data).count()

        println("tp:"+tp)
        println("fn:"+fn)
        println("fp:"+fp)

        print("Precision:"+tp.toDouble/(tp+fp))
        print("Recall:"+ tp.toDouble/(tp+fn))
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")



  }







  def minhash(user_list: List[Int]): ListBuffer[Int] = {
    var minhash_list = ListBuffer.empty[Int]
    for(i <- 0 until hash_num){
      var list_after_hash = ListBuffer.empty[Int]
      for(num <- user_list){


        val new_index = (a_list(i) * num + b_list(i)) % m_list(i)

        list_after_hash += new_index
      }

      minhash_list+= list_after_hash.min
    }
    return minhash_list
  }


}
