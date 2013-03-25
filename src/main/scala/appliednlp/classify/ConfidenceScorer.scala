package appliednlp.classify
import scala.collection.mutable


/**
 * An application that takes a gold labeled file and a file containing
 * predictions, and then computes the accuracy for the top-third most
 * confident instances and the accuracy for the bottom-third (least 
 * confident instances).
 */
object ConfidenceScorer {

  def main(args: Array[String]) {

  	val groundTruthMap = mutable.Map[Int,String]()

  	val confidenceScores = io.Source.fromFile(args(0)).getLines.toIndexedSeq.zipWithIndex

  	confidenceScores.foreach{ x=>

  		val groundTruth = x._1.takeRight(2)
  		val index = x._2
  		groundTruthMap(index) = groundTruth.trim
  	}

  	val sortedPredictions = io.Source.fromFile(args(1))
  	.getLines
  	.toIndexedSeq
  	.zipWithIndex
  	.map(x=> (x._2,x._1.split(" ").toIndexedSeq))
  	.sortBy(x=>x._2(1)).reverse


  	val highEnd = (sortedPredictions.length*1.0/3).ceil.toInt 
  	val medEnd = (sortedPredictions.length*2.0/3).ceil.toInt 



  	val highConfidence =  sortedPredictions.slice(0,highEnd)
  	val medConfidence = sortedPredictions.slice(highEnd,medEnd)
  	val lowConfidence = sortedPredictions.slice(medEnd,sortedPredictions.length)



  	val highAccuracy = 1.0*highConfidence.count(x=> groundTruthMap(x._1).equals(x._2(0))) / highConfidence.length
  	val medAccuracy = 1.0*medConfidence.count(x=> groundTruthMap(x._1).equals(x._2(0))) / medConfidence.length
  	val lowAccuracy = 1.0*lowConfidence.count(x=> groundTruthMap(x._1).equals(x._2(0))) / lowConfidence.length

  	println("High Confidence Accuracy :" + highAccuracy*100)
  	println("Medium Confidence Accuracy :" + medAccuracy*100)
  	println("Low Confidence Accuracy :" + lowAccuracy*100)







  	//val lines = io.Source.fromFile("") 

  }  

}
