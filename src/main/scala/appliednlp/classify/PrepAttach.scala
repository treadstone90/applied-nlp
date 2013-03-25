package appliednlp.classify

import nak.core.AttrVal
import chalk.lang.eng.PorterStemmer
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import edu.mit.jwi.morph.WordnetStemmer;
import java.io.File;
import scala.collection.mutable
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.TaggedWord;
import scala.collection.JavaConversions._

/**
 * An object that sets up the configuration for command-line options using
 * Scallop and returns the options, ready for use.
 */
object PpaFeaturesOpts {

  import org.rogach.scallop._
  
  def apply(args: Array[String]) = new ScallopConf(args) {
    banner("""
For usage see below:
	     """)
    val help = opt[Boolean]("help", noshort = true, descr = "Show this message")
    val verbose = opt[Boolean]("verbose")
    val bitstringsSource = opt[String]("bitstrings", descr = "File containing bitstrings")
    val extendedFeatures = opt[Boolean]("extended",short='e', descr="Use extended features.")
    val inputFile = trailArg[String]("inputfile", descr = "Input file to create features from.")
  }
}


/**
 * An application for extracting features from the PPA native format for 
 * classification.
 */
object PpaFeatures {
  /**
   * The main method -- do the work. Don't change it.
   */
  def main(args: Array[String]) {

    // Parse and get the command-line options
    val opts = PpaFeaturesOpts(args)
   
    val inputFile = opts.inputFile()

    val bitstrings = opts.bitstringsSource.get match {
      case Some(bitstringsSource) =>
        io.Source.fromFile(bitstringsSource).getLines.map { line =>
          val Array(word, bitstring) = line.split("\\s+")
          (word -> BitVector(bitstring))
        }.toMap

      case None => new collection.immutable.HashMap[String, BitVector]()
    }

    val featureExtractor =
      if (opts.extendedFeatures()) new ExtendedFeatureExtractor(bitstrings)
      else BasicFeatureExtractor

    io.Source.fromFile(inputFile).getLines.foreach { line =>
      val Array(id, verb, noun, prep, prepObj, attach) = line.split(" ")
      val features = featureExtractor(verb, noun, prep, prepObj)
      println(features.map(_.toString).mkString(",") + "," + attach)
    }

  }

}

/**
 * A trait for classes that can extract features from the information in
 * the PPA files.
 */
trait FeatureExtractor {
  
  /**
   * Given the verb, noun, preposition, and prepositional object,
   * create a set of AttrVal objects. (A "feature" is an attribute with a
   * value.) 
   */
  def apply(verb: String, noun: String, prep: String, prepObj: String): Iterable[AttrVal]
}

/**
 * The simplest feature extractor: each word gets a feature, where the 
 * attribute is the type of the word. 
 */
object BasicFeatureExtractor extends FeatureExtractor {

  override def apply(
    verb: String, noun: String, prep: String, prepObj: String): Iterable[AttrVal] = {
    List(
      AttrVal("verb", verb),
      AttrVal("noun", noun),
      AttrVal("prep", prep),
      AttrVal("prep_obj", prepObj))
  }

}

/**
 * An extended feature extractor. It is your job to fill this out further.
 */
class ExtendedFeatureExtractor(bitvectors: Map[String, BitVector])
  extends FeatureExtractor {

  lazy val stemmer = new PorterStemmer
  lazy val wordnetDict = new Dictionary (new File("/home/akarthik/WordNet-2.0/dict/"));
  wordnetDict.open


  val wordDomainMap = mutable.Map[String,String]().withDefault(x=>"UNK")
  val suffixes = IndexedSeq("able","ible","al","ial","ed","en","er","er","est","ful","ic","ing","ion",
    "tion","ation","ition","ity","ty","ive","ative","itive","less","ly","ment","ness","ous","eous","ious","s","es","y")
  .map(x=> (x,x.length))
  .sortBy(x=>x._2)
  .reverse
  .unzip._1


  {
    io.Source.fromFile("wndomain/wn-domains-3.2-20070223").getLines.foreach { line =>
      val tokens = line.split("\\s+")
      if(tokens(0).endsWith("-n"))
        wordDomainMap(tokens(0).dropRight(2)) = tokens(1)
    }
  }

  val tagger:MaxentTagger = new MaxentTagger("POSTagger/faster.tagger");

  override def apply(
    verb: String, noun: String, prep: String, prepObj: String): Iterable[AttrVal] = {

    // Use the basic feature extractor to get the basic features (no need to 
    // duplicate effort and specify it again).
    val basicFeatures = BasicFeatureExtractor(verb, noun, prep, prepObj)
    val lengthFeatures = getLength(verb,prep)
    val capitalizeFeautures = getCapFeatures(noun,prepObj)
    val NumberFeatures = getNumFeatures(noun,prepObj) 
    val DomainFeatures = getDomainFeatures(noun, prepObj);
    val stemmedFeatures = getStemmedFeatures(verb, noun,prepObj)
    val suffixFeatures = getSuffixFeatures(verb, noun, prepObj)
    val basicCombination = getbasicCombination(verb,noun,prep,prepObj)
    val stemmedCombination = getStemmedCombination(verb,noun,prep,prepObj)
    val bitstringFeature = getBitStringFeatures(verb,noun,prep,prepObj) 
    val verbTypeFeature = getVerbType(verb,noun,prep,prepObj);

    basicFeatures ++ lengthFeatures ++ capitalizeFeautures ++ NumberFeatures ++ DomainFeatures ++ suffixFeatures ++ stemmedFeatures ++ 
    stemmedCombination ++ basicCombination ++  bitstringFeature ++ verbTypeFeature
  }

  def getVerbType(verb:String,noun:String,prep:String,prepObj:String) :Iterable[AttrVal] =
  {  
    val hasWordline = List(verb,noun,prep,prepObj).map(x=> new Word(x))
    val POSTagged = tagger.tagSentence(hasWordline).toIndexedSeq
    val verbPOS = (POSTagged(0).toString.split("/"))(1)

    if(verbPOS.startsWith("VB")) List(AttrVal("verb_type",verbPOS))  else List(AttrVal("verb_type","VB"))
  }

  def getBitStringFeatures(verb:String,noun:String,prep:String,prepObj:String) :Iterable[AttrVal] ={
      val prepFeatures = getBitStringFeature(prep,"prep",IndexedSeq(8,12,14,16,20))
      val nounFeatures = getBitStringFeature(noun,"noun",IndexedSeq(8,12,14,16,20))
      val verbFeatures = getBitStringFeature(verb,"verb",IndexedSeq(8,12,14,16,20))
      val prepObjFeatures = getBitStringFeature(prepObj,"prepObj",IndexedSeq(8,12,14,16,20))

      prepFeatures ++ nounFeatures ++ verbFeatures ++ prepFeatures
  }

  def getBitStringFeature(word:String,POS:String,prefixLength:IndexedSeq[Int]) =
  {
      prefixLength.map(x=> AttrVal(POS+"_"+x.toString+"_"+"prefix",bitvectors(word).keepTopBits(x).toString)).toList
  }

  def getStemmedCombination(verb:String,noun:String,prep:String,prepObj:String) :Iterable[AttrVal] ={
    List(
      AttrVal("verb_noun_st",stemmer(verb)+"_"+stemmer(noun)),
      AttrVal("noun_prep_st",stemmer(noun)+"_"+stemmer(prep)),
      AttrVal("prep_prepObj_st",stemmer(prep)+"_"+stemmer(prepObj)),
      AttrVal("noun_prepObj_st",stemmer(noun)+"_"+stemmer(prepObj))
      )
  }

  def getbasicCombination(verb:String,noun:String,prep:String,prepObj:String) :Iterable[AttrVal] ={
    List(
      AttrVal("verb_noun", verb+"_"+noun),
      AttrVal("noun_prep", noun+"_"+prep),
      AttrVal("prep_prepObj", prep+"_"+prepObj),
      AttrVal("noun_prepObj", noun+"_"+prepObj),
      AttrVal("noun_prep_prepObj", noun+"_"+prep+"_"+prepObj)
      )
  }

  def getSuffixFeatures(verb:String,noun:String,prepObj:String):Iterable[AttrVal] =
  {
    val nounSuffixes =getSuffixes(noun);
    val nounSuffixFeature= if(nounSuffixes.length == 0) "NA" else nounSuffixes(0)  

    val verbSuffixes =getSuffixes(verb);
    val verbSuffixFeature= if(verbSuffixes.length == 0) "NA" else verbSuffixes(0)  

    val prepObjSuffixes =getSuffixes(prepObj);
    val prepObjSuffixFeature= if(prepObjSuffixes.length == 0) "NA" else prepObjSuffixes(0)  

    List(
      AttrVal("noun_suffix",nounSuffixFeature),
      AttrVal("verb_suffix",verbSuffixFeature),
      AttrVal("prepObj_suffix",prepObjSuffixFeature)
      )
  }

  def getSuffixes(str:String) = suffixes.filter(x=> str.endsWith(x) == true)

  def getStemmedFeatures(verb:String, noun:String, prepObj:String)=
  {
    List(
      AttrVal("verb_stem",stemmer(verb)),
      AttrVal("noun_stem",stemmer(noun)),
      AttrVal("prepObj_stem",stemmer(prepObj))
      )
  }

  def getDomainFeatures(noun:String, prepObj:String): Iterable[AttrVal] ={
    
    val noun_domain = getDomain(noun)
    val prepObj_domain = getDomain(prepObj)

    List(
      AttrVal("noun_domain",noun_domain),
      AttrVal("prepObj_domain",prepObj_domain),
      AttrVal("noun+prepObj_domain" , noun_domain+"_"+prepObj_domain)
      )
  }
  def getDomain(str:String)={
   // println(wordnetDict);
    val stems = new WordnetStemmer(wordnetDict).findStems(str,POS.NOUN);

    val x = if(stems.size() ==0) str else stems.get(0)
    val idxWord = wordnetDict.getIndexWord(x, POS.NOUN);
    if(idxWord == null) "UNK"
    else
    {
    val  wordID = idxWord.getWordIDs().get(0) ;
    val word = wordnetDict.getWord(wordID);
    val offset = word.getSynset().getOffset.toString

    val domain = wordDomainMap("0"*(8-offset.length) + offset);
    domain
    }
 
  }


  def getNumFeatures(noun:String,prepObj:String):Iterable[AttrVal] ={
    List(
      AttrVal("num_Noun",matchNumForm(noun)),
      AttrVal("num_prepObj",matchNumForm(prepObj))
      )
  }

  def matchNumForm(x:String)={
    val numMatch = """\d.*?\b""".r
    x match { 
      case numMatch() => "true"
      case _ => "false"
    }
  }

  def getLength(verb:String, prep:String):Iterable[AttrVal] ={
    List(
      AttrVal("len_prep",prep.length.toString),
      AttrVal("len_verb",verb.length.toString)
      )
  }

  def getCapFeatures(noun:String, prep:String):Iterable[AttrVal] ={
    List(
      AttrVal("cap_Noun",matchCapForm(noun)),
      AttrVal("cap_prepObj", matchCapForm(prep))
      )
  }

  def matchCapForm(x:String)= {
    val fullUpper = """[A-Z]+""".r
    val startsUpper = """[A-Z]\w+""".r

    x match {
      case fullUpper() => "X"
      case startsUpper()=> "Xx"
      case _ => "false" 
    }
    
  }


}

/**
 * This is an entirely cruddy, slow implementation of a bit vector,
 * not using any bitwise ops, etc., but it should suffice for this problem.
 *
 * And, yes, we are using Ints where it could be Booleans, and we could have
 * the wrong values in there, but this keeps it easy, and again, is sufficient
 * for this problem.
 * 
 * Feel free to add more capability to this if it helps you create better
 * features.
 */
class BitVector(bits: IndexedSeq[Int]) {

  /**
   * Get the bit value at the given index.
   */
  def apply(index: Int) = bits(index)

  /**
   * Get the integer value of the bits
   */
  lazy val toInt = Integer.parseInt(bits.mkString, 2)

  /**
   *  Keep the top bits up to the given index, and then make the remaining bits
   *  zero.
   */
  def keepTopBits(index: Int) =
    new BitVector(bits.take(index) ++ Vector.fill(bits.length - index)(0))

  /**
   * Concatenate the bits together.
   */
  override def toString = bits.mkString
}

/**
 * Companion object to the BitVector class.
 */
object BitVector {

  /**
   * Create a bit vector from a string of zeros and ones.
   */
  def apply(bitstring: String) =
    new BitVector(bitstring.split("").drop(1).map(_.toInt).toIndexedSeq)
}



