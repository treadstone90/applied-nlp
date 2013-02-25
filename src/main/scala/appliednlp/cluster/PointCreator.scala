package appliednlp.cluster

import nak.cluster._
import nak.util.CollectionUtil._
import chalk.util.SimpleTokenizer
import chalk.tools.sentdetect._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.io.Source
import java.io.FileInputStream
import scala.collection.mutable


/**
 *  Read data and produce data points and their features.
 *
 *  @param filename the name of the file containing the data
 *  @return a triple, the first element of which is a sequence of id's
 *     (unique for each row / data point), the second element is the sequence
 *     of (known) cluster labels, and the third of which is the sequence of
 *     Points to be clustered.
 */
trait PointCreator extends (String => Iterator[(String,String,Point)])

/**
 * Read data in the standard format for use with k-means.
 */
object DirectCreator extends PointCreator {

 def apply(filename: String) ={
   val inputLines =Source.fromFile(filename).getLines.toList;

   inputLines.map(x=> x.split("\\s+").toIndexedSeq).map(x=> (x(0),x(1),new Point(IndexedSeq(x(2).toDouble,x(3).toDouble)))).toIterator;

 }

}


/**
 * A standalone object with a main method for converting the achieve.dat rows
 * into a format suitable for input to RunKmeans.
 */
object SchoolsCreator extends PointCreator {

sealed trait Grade
object FourthGrade extends Grade
object SixthGrade extends Grade

//List[(String,String,Point)]().toIterator
  def apply(filename: String) = {
      val input =Source.fromFile(filename).getLines.toList.map(x=> x.split("\\s+").toIndexedSeq)
      val fourthGraders = input.map(x=> (getName(x,"4"),"4",new Point(IndexedSeq(x(x.length-4).toDouble,x(x.length-3).toDouble))))
      val sixthGraders = input.map(x=> (getName(x,"6"),"6",new Point(IndexedSeq(x(x.length-2).toDouble,x(x.length-1).toDouble))))
        (fourthGraders++sixthGraders).toIterator
  }

  def getName(x:IndexedSeq[String],grade:String)= {
    
    val name = if(x.length == 5) x(0) else {
      x.slice(0,x.length-5+1).mkString("")
    }
    grade match {
      case "4" => name.trim+"_4th"
      case "6"=> name.trim+"_6th"
  
  }

}
}

/**
 * A standalone object with a main method for converting the birth.dat rows
 * into a format suitable for input to RunKmeans.
 */

object CountriesCreator extends PointCreator {

  def apply(filename: String) = {
    val input =Source.fromFile(filename).getLines.toList.map(x=> x.split("\\s+").toIndexedSeq)

    val countriesData = input
    .map(x=> ( 
      x.slice(0,x.length-2).mkString("_") , 
      "1",
      new Point(IndexedSeq(x(x.length-1).toDouble,x(x.length-2).toDouble))))

    countriesData.toIterator
  }
}

/**
 * A class that converts the raw Federalist
 * papers into rows with a format suitable for input to Cluster. As part of
 * this, it must also perform feature extraction, converting the texts into
 * sets of values for each feature (such as a word count or relative
 * frequency).
 */

 

class FederalistCreator(simple: Boolean = false) extends PointCreator {

  def apply(filename: String) =  
  {
    val federalistData = FederalistArticleExtractor(filename)
    val texts = federalistData.map(x=> x("text")).toIndexedSeq
    val ids = federalistData.map(x=> x("id")).toIndexedSeq
    val author = federalistData.map(x=> x("author")).toIndexedSeq

    val dataPoints = if(simple == true) extractSimple(texts) else extractFull(texts)

      
    val dataInput = for(i<- 0 to federalistData.length-1) yield (ids(i),author(i),dataPoints(i));
   
    dataInput.toIterator

  }

  /**
   * Given the text of an article, compute the frequency of "the", "people"
   * and "which" and return a Point per article that has the frequency of
   * "the" as the value of the first dimension, the frequency of "people"
   * for the second, and the frequency of "which" for the third.
   *
   * @param texts A sequence of Strings, each of which is the text extracted
   *              for an article (i.e. the "text" field produced by
   *              FederalistArticleExtractor).
   */
  def extractSimple(texts: IndexedSeq[String]): IndexedSeq[Point] = 
  {
    // texts is List of Strings where each String is the text

    println("Simple");
    val reducedArticle: IndexedSeq[IndexedSeq[String]]=texts.map(x=> SimpleTokenizer(x.toLowerCase).
      filter(x=> x.equals("the")|x.equals("which")|x.equals("people")))

    val articleMaps:IndexedSeq[Map[String,Int]]=reducedArticle.map(x=> x.groupBy(x=>x).mapValues(x=>x.length).withDefault(x=>0))

    articleMaps.map(x=> new Point(IndexedSeq(x("the").toDouble, x("which").toDouble,x("people").toDouble)))

  }

  /**
   * Given the text of an article, extract features as best you can to try to
   * get good alignment of the produced clusters with the known authors.
   *
   * @param texts A sequence of Strings, each of which is the text extracted
   *              for an article (i.e. the "text" field produced by
   *              FederalistArticleExtractor).
   */
  def extractFull(texts: IndexedSeq[String]): IndexedSeq[Point] = 
  {
      // firsr lets get the ratio of capitalized words
      val sdetector = new SentenceDetectorME(new SentenceModel(new FileInputStream("/home/akarthik/en-sent.bin")))
      
      val articlesLowerCase:IndexedSeq[IndexedSeq[String]] = texts.map(x=> SimpleTokenizer(x.toLowerCase))
      val textSets=articlesLowerCase.map(x=> x.toSet)
      val articlesNormal:IndexedSeq[IndexedSeq[String]] = texts.map(x=> SimpleTokenizer(x))
      val fWords = io.Source.fromFile("/home/akarthik/fWords.txt").mkString.split("\\s+").toIndexedSeq
      val filteredFWords = getPotentialFWords(fWords,textSets);
      val articlesWordCount : IndexedSeq[Map[String,Double]] = getWordCount(articlesLowerCase)  // all lower case


      val articlefeatureMaps = new Array[mutable.Map[String,IndexedSeq[Double]]](85)

      for(i<- 0 to articlefeatureMaps.length-1) 
      {
        articlefeatureMaps(i) = mutable.Map[String,IndexedSeq[Double] ]().withDefault(x=>IndexedSeq(0.0))
      }


      val vocabulary = articlesLowerCase.flatten.toSet;
      val corpusWordCount = articlesLowerCase.flatten.groupBy(x=>x).mapValues(x=> x.length)

      val documentFrequencyMap:Map[String,Double] = documentFrequency(vocabulary,textSets)

      

      //val selectedvocabulary = dffeatureSelection(documentFrequencyMap,60) -- filteredFWords;

      val selectedVocabulary = entropyFeatureSelection(vocabulary,articlesWordCount,corpusWordCount).map(x=> x._1).toSet -- filteredFWords
      
     /* for(i<- 0 to articlefeatureMaps.length-1)
      {
         println(tfidf(articlesLowerCase(i), documentFrequencyMap, 
              articlesWordCount(i), selectedVocabulary.toIndexedSeq))
      }
*/
      
      for(i<- 0 to articlefeatureMaps.length-1)
      {
        articlefeatureMaps(i)("avesentlength") = IndexedSeq(averageSentLength(texts(i), sdetector))
        articlefeatureMaps(i)("capRatio") = IndexedSeq(getCapitalWordRatio(articlesNormal(i)))
        articlefeatureMaps(i)("avePunct") = IndexedSeq(avePunctInSentence(texts(i), sdetector))
        articlefeatureMaps(i)("numsent") =  IndexedSeq(numberOfSentences(texts(i), sdetector))
        articlefeatureMaps(i)("fRatio") =  functionWordsRatio(texts(i) ,filteredFWords)
        articlefeatureMaps(i)("vrichness") = IndexedSeq(vocabRichness(texts(i) , vocabulary -- fWords))
        articlefeatureMaps(i)("tfidf") = tfidf(articlesLowerCase(i), documentFrequencyMap, 
              articlesWordCount(i), selectedVocabulary.toIndexedSeq)
      }


      val points = for(i<- 0 to articlefeatureMaps.length-1) 

         yield new Point( IndexedSeq(
                  articlefeatureMaps(i)("avesentlength"),
                  //articlefeatureMaps(i)("capRatio"),
                  articlefeatureMaps(i)("avePunct"),
                 //articlefeatureMaps(i)("numsent"),
                  //articlefeatureMaps(i)("vrichness"),
                  articlefeatureMaps(i)("fRatio"),
                 
                 articlefeatureMaps(i)("tfidf")
                  ).flatten)
      
         points.toIndexedSeq

      //Vector[Point]()
   
  }
  def avePunctInSentence(text:String, sdetector:SentenceDetectorME) ={

    val numSent = numberOfSentences(text,sdetector);

    val numberOfPunct=SimpleTokenizer(text).count(x=> x.matches("""[,;\.?]"""));

    1.0*numberOfPunct/numSent

  }

  def entropyFeatureSelection(vocab:Set[String], termFrequencyMap:IndexedSeq[Map[String,Double]],
    corpusWordCount:Map[String,Int])={


    val wordEntropies= vocab.map(word=> (word, (1+ entropy(word,termFrequencyMap,corpusWordCount(word))/(Math.log(85)))))
    .toIndexedSeq.sortBy(x=> x._2).reverse
    
    wordEntropies.filter(x=> x._2 < 0.15).slice(0,10)


  }

  def entropy(word:String, termFrequencyMap:IndexedSeq[Map[String,Double]],corpusCount:Int)={
    termFrequencyMap.map(x=> x(word)).map(x=> if(x>0.0) (x/corpusCount)*Math.log(x/corpusCount) else 0).sum
  }

  def tfidf(article:IndexedSeq[String],documentFrequencyMap:Map[String,Double],
    termFrequencyMap:Map[String,Double],
    selectedvocabulary:IndexedSeq[String]) = {

  selectedvocabulary.map(x=> termFrequencyMap(x) *
     Math.log(1.0*85/(1+ documentFrequencyMap(x))));

  }



  def dffeatureSelection(dfMap:Map[String,Double],d:Int):Set[String] = 
  {
    dfMap.keys.filter(x=> dfMap(x) >=d).toSet
  }

  def getCapitalWordRatio(article: IndexedSeq[String]) = {
    article.count(x=> x.matches("[A-Z]+") && x.length > 1)/(article.length*1.0) 
  }

  def getWordCount(articles : IndexedSeq[IndexedSeq[String]]) : IndexedSeq[Map[String,Double]]={
    articles.map(article=> article.groupBy(x=>x).mapValues(x=>x.length.toDouble).withDefault(x=>0.0))
  }

  def functionWordsRatio(text:String,fWords:IndexedSeq[String]): IndexedSeq[Double]= 
  {
    val articleTokens = SimpleTokenizer(text.toLowerCase)
    val articleWordCount = articleTokens.groupBy(x=> x).mapValues(x=> x.length).withDefault(x=>0)
    fWords.map(x=> articleWordCount(x).toDouble/articleTokens.length)

  }

  def vocabRichness(text:String,fWords:Set[String]):Double ={
    (SimpleTokenizer(text.toLowerCase).toSet -- fWords).size
  }
  def numberOfSentences(text:String, sdetector:SentenceDetectorME) : Double ={
    sdetector.sentDetect(text).length.toDouble
  }

  def averageSentLength(text:String, sdetector:SentenceDetectorME) : Double = {
    (SimpleTokenizer(text).length*1.0)/(numberOfSentences(text,sdetector))  
  }

  def termFrequency(words:IndexedSeq[String], document: IndexedSeq[String]): IndexedSeq[Double]=
  {
    val documentMap = document.groupBy(x=>x).mapValues(x=> x.length).withDefault(x=>0)

    words.map(x=> documentMap(x).toDouble).toIndexedSeq
  }

  def getPotentialFWords(Fwords:IndexedSeq[String],textSets : IndexedSeq[Set[String]]):IndexedSeq[String] = {
    Fwords.filter(x=> documentFrequency(x,textSets) > 50)  


  }


  def documentFrequency(words:Set[String], texts: IndexedSeq[Set[String]]) = {

      words.map(word=> (word,texts.filter(x=> x.contains(word)).length.toDouble)).toMap
  }
  def documentFrequency(word:String, texts: IndexedSeq[Set[String]])= texts.filter(x=> x.contains(word)).length



}

object FederalistArticleExtractor {
  /**
   * A method that takes the raw Federalist papers input and extracts each
   * article into a structured format.
   *
   * @param filename The filename containing the Federalist papers.
   * @return A sequence of Maps (one per article) from attributes (like
   *         "title", "id", and "text") to their values for each article.
   */
  def apply(filename: String): IndexedSeq[Map[String, String]] = {

    // Regex to identify the text portion of a document.
    val JustTextRE = (
      """(?s)\*\*\* START OF THIS PROJECT GUTENBERG.+""" +
      """\*\*\*(.+)\*\*\* END OF THIS PROJECT GUTENBERG""").r

    // Regex to capture different parts of each article.
    val ArticleRE = (
      """(?s)(\d+)\n+""" + // The article number.
      """(.+?)\n+""" + // The title (note non-greedy match).
      """((?:(?:For|From)[^\n]+?)?)\s+""" + // The publication venue (optional).
      """((?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday).+\d\d\d\d\.)?)\n+""" + // The date (optional).
      """((?:MAD|HAM|JAY).+?)\n+""" + // The author(s).
      """(To the [^\n]+)""" + // The addressee.
      """(.+)""" // The text.
      ).r

    val book = io.Source.fromFile(filename).mkString
    val text = JustTextRE.findAllIn(book).matchData.next.group(1)
    val rawArticles = text.split("FEDERALIST.? No. ")

    // Use the regular expression to parse the articles.
    val allArticles = rawArticles.flatMap {
      case ArticleRE(id, title, venue, date, author, addressee, text) =>
        Some(Map("id" -> id.trim,
          "title" -> title.replaceAll("\\n+", " ").trim,
          "venue" -> venue.replaceAll("\\n+", " ").trim,
          "date" -> date.replaceAll("\\n+", " ").trim,
          "author" -> author.replaceAll("\\n+", " ").trim,
          "addressee" -> addressee.trim,
          "text" -> text.trim))

      case _ => None
    }.toIndexedSeq

    // Get rid of article 71, which is a duplicate, and return the rest.
    allArticles.take(70) ++ allArticles.slice(71, allArticles.length)
  }

}