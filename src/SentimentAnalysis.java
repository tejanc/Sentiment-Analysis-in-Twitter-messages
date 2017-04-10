import java.io.File;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class SentimentAnalysis {

	private static final String FILE = "C:\\Users\\Tejan\\Dropbox\\workspace\\Sentiment Analysis in Twitter Messages\\src\\semeval_twitter_data.txt";

	public static void main (String[] args) {

		//createARFF(FILE);
		
		try {
			runSentimentAnalysis();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void createARFF(String file) {
		
		tweetTokenizer();
	}
	
	/*
	 * Removes all html tags and attribute e.g. /<[^>]+>/
	 * All URLs i.e. http or www are removed
	 * The first character in Twitter usernames and hash tags e.g @ and # are removed.
	 */
	public static void tweetTokenizer() {
		
	}

	public static void runSentimentAnalysis() throws Exception {
		ArffLoader loader = new ArffLoader();
	    loader.setFile(new File("src/semeval_twitter_data.arff"));
	    Instances breader = loader.getDataSet();
	    
		Instances train = new Instances (breader);
		train.setClassIndex(train.numAttributes() - 1);
				
		StringToWordVector filter = new StringToWordVector();
	    filter.setInputFormat(train);
	    Instances dataFiltered = Filter.useFilter(train, filter);
		
		NaiveBayes nB = new NaiveBayes();
		try {
			nB.buildClassifier(dataFiltered);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} // not necessary as we are doing cross validation.
		Evaluation eval = null;
		try {
			eval = new Evaluation(train);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			eval.crossValidateModel(nB, dataFiltered, 10, new Random(1));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println(eval.toSummaryString("\nResults\n=======\n", true));
		System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));		
	}
	

}