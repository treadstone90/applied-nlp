package appliednlp.classify
import scala.io.Source
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.ling.Word;
import java.io.IOException;
import java.io.StringReader;
import edu.stanford.nlp.process.PTBTokenizer.PTBTokenizerFactory;
import scala.collection.JavaConversions._
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.TaggedWord;
import sentiWN.SentiWordNet
import scala.collection.mutable.ArrayBuffer;
import java.io.FileWriter

/**
*An application that takes as input the train and test files
* and then produces the training file in the format required by 
* nak --classify. I guess the nak classification format is the same 
* as the Liblinear format , so I will write this in the lib-linear format

*/

// Usage
// TextClassify --train <trainFile>  --split number --test <testFile> --split number --SWN <SWNpath> --POS <POSTAGpath> 
object TextClassify {
	def main(args:Array[String]) {

		val tagger:MaxentTagger = new MaxentTagger(args(5));
		val trainLines = Source.fromFile(args(0),"ISO-8859-1").getLines.toIndexedSeq
		val testLines = Source.fromFile(args(2),"ISO-8859-1").getLines.toIndexedSeq
		val swn = new SentiWordNet(args(4));
		val trainSplit = args(1).toInt
		val testSplit = args(3).toInt



		val vocabulary =scala.collection.mutable.HashSet[String]()
		val stopwords = Source.fromFile(args(6)).getLines.toSet
		
		

		val trainLinesTokenized = getTokenizedLines(trainLines)
		val testLinesTokenized = getTokenizedLines(testLines)

		
		trainLinesTokenized.foreach{ x=>
			x.foreach(w=> if(!stopwords.contains(w)) vocabulary.add(w))
			}

		
		val vocabList = vocabulary.toIndexedSeq;
		val wordIndexMap = scala.collection.mutable.Map[String,Int]().withDefault(x=> -1)


		vocabList.zipWithIndex.foreach{x=>
			val index = x._2
			val word = x._1
			wordIndexMap(word) = index+1
		}

		val maxWordIndex = wordIndexMap.toIndexedSeq.maxBy(x=>x._2)._2;

	
		
		val trainTDMatrix = getTDMatrix(wordIndexMap,vocabulary,trainLinesTokenized,tagger,swn,maxWordIndex);
		val testTDMatrix = getTDMatrix(wordIndexMap,vocabulary,testLinesTokenized,tagger,swn,maxWordIndex);

		writeTDMatrix(trainTDMatrix,"/home/akarthik/Dropbox/Utexas/Spring_13/applied_nlp/nak/data/classify/rotten_imdb/train.features",trainSplit)
		writeTDMatrix(testTDMatrix,"/home/akarthik/Dropbox/Utexas/Spring_13/applied_nlp/nak/data/classify/rotten_imdb/test.features",testSplit)


		// word Count for the words
	}

	def writeTDMatrix(TDMatrix: ArrayBuffer[ArrayBuffer[(Int,Double)]], fileName : String,split:Int)
	{
		val wr:FileWriter  = new FileWriter(fileName)

		TDMatrix.zipWithIndex.foreach{ case(document,index)=>

			val classLabel = if(index< split) "+1" else "-1"
			wr.write(classLabel + " ")
			document.foreach{feature=>
				wr.write(feature._1+":"+feature._2)
				wr.write(" ")
			}
			
			wr.write("\n");
		}

		wr.close
	}

	def getTDMatrix(wordIndexMap:scala.collection.mutable.Map[String,Int],vocabulary: scala.collection.mutable.HashSet[String], linesTokenized: IndexedSeq[IndexedSeq[String]],tagger:MaxentTagger,swn:SentiWordNet,maxWordIndex:Int) ={
		
		val idfMap = calculateIDF(vocabulary,linesTokenized)
		val TDMatrix = ArrayBuffer[ArrayBuffer[(Int,Double)]]()

		val deltaIDFMap = calculateDeltaIDF(vocabulary,linesTokenized)


		linesTokenized.foreach{x=>
			val hasWordline = x.map(x=> new Word(x))
			val featureVector = ArrayBuffer[(Int,Double)]()
			
			val POSTagged =  tagger.tagSentence(hasWordline).toIndexedSeq
			val adjAdvFeature = getAdjAdvFeature(POSTagged);
			val SWNScore = getSWNScore(POSTagged,swn);
			val TFIDF= getTFIDF(vocabulary,x,deltaIDFMap);
			
			TFIDF.foreach { case(key,value)=>
				val index = wordIndexMap(key)
				val tuple = (index,value)
				featureVector+= tuple
			}

			val tupleadjadv = (maxWordIndex+1, adjAdvFeature)  
			val tupleSWN = 	(maxWordIndex+2, SWNScore) 
			
			featureVector+= tupleadjadv
			featureVector+= tupleSWN

			val sortedfeatureVector = featureVector.sortBy(x=> x._1)

			TDMatrix += sortedfeatureVector
			
		}

		println(TDMatrix.length)
		TDMatrix


	}

	def getTokenizedLines(lines:IndexedSeq[String]) ={

		lines.map { x=>
			val tokens = PTBTokenizer.factory().getTokenizer(new StringReader(x)).tokenize();
			tokens.map(x=> x.toString.trim).filter(x=> !x.matches("[\\p{Punct}\\s]+")).toIndexedSeq
		}
	}

	def getSWNScore(taggedWords: IndexedSeq[TaggedWord],swn:SentiWordNet) ={
		val nounsVerbs = taggedWords.map(x=> x.toString.split("/").toIndexedSeq).
					filter(x=> x(1).startsWith("NN") || x(1).startsWith("VB") || x(1).startsWith("JJ") || x(1).startsWith("VB"))
		//println(nounsVerbs)

		if(nounsVerbs.length ==0)
		0.5
		else
		1.0*nounsVerbs.map(x=> Math.abs(swn.extractScore(x(0),x(1)(0).toString.toLowerCase)).toDouble).sum/nounsVerbs.length
	}

	def getTFIDF(vocabulary:scala.collection.mutable.HashSet[String],lineTokenized:IndexedSeq[String],idfMap : scala.collection.Map[String,Double]) ={
		val wordCount = lineTokenized.groupBy(x=>x).mapValues(x=>x.length)
		val sentenceVocab = vocabulary & lineTokenized.toSet


		sentenceVocab.map(x=> (x,wordCount(x)*idfMap.getOrElse(x,0.0))).toMap
	}
	

	def getAdjAdvFeature(taggedWords:IndexedSeq[TaggedWord]) ={

		//val taggedWords = tagger.tagSentence(tokens).toIndexedSeq
		val countAdjAdv = taggedWords.map(x=> (x.toString.split("/"))(1))
		.count(x=> (x.startsWith("JJ") || x.startsWith("RB")))

		1.0*countAdjAdv/taggedWords.size
	}

	def calculateIDF(vocabulary:scala.collection.mutable.HashSet[String], linesTokenized: IndexedSeq[IndexedSeq[String]])=
	{
		//now keep in mind that the first 2500 documents are , one class and the rest are the other class
		// iDf is calculated per words, so do it per word., as this makes more sense
		val IDFMap = scala.collection.mutable.Map[String,Double]().withDefault(x=> 0.0);
		val linesHash = linesTokenized.map(x=> x.toSet)

		vocabulary.foreach{x=>
			val word = x
			linesHash.foreach{ line =>
				
				if(line.contains(word)) 
					IDFMap(word)+=1.0
			}
		}

		IDFMap.mapValues(x=> Math.log(linesTokenized.length/(1+x)))


		/*linesHash.foreach{ line=>
			line.foreach(word =>IDFMap(word) +=1.0)
		}*/

	}

	def calculateDeltaIDF(vocabulary:scala.collection.mutable.HashSet[String], linesTokenized: IndexedSeq[IndexedSeq[String]])=
	{
		//now keep in mind that the first 2500 documents are , one class and the rest are the other class
		// iDf is calculated per words, so do it per word., as this makes more sense
		val IDFMap = scala.collection.mutable.Map[String,Double]().withDefault(x=> 0.0);
		val linesHashClass1 = linesTokenized.slice(0,2500).map(x=> x.toSet)
		val linesHashClass2 = linesTokenized.slice(2500,5000).map(x=> x.toSet)


		vocabulary.foreach{x=>
			val word = x
			val class1Count = linesHashClass1.count(x=> x.contains(word))
			val class2Count = linesHashClass2.count(x=> x.contains(word))
			IDFMap(word) = 1.0*(class2Count*linesHashClass1.length)/(class1Count*linesHashClass2.length)
		}

		IDFMap.mapValues(x=> Math.log(x))


		/*linesHash.foreach{ line=>
			line.foreach(word =>IDFMap(word) +=1.0)
		}*/

	}


}
