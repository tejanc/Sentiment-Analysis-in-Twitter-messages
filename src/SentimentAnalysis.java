import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import javax.management.Attribute;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.FastVector;
import weka.core.Instances;

public class SentimentAnalysis {

	private static final String FILE = "C:\\Users\\Tejan\\Dropbox\\workspace\\Sentiment Analysis in Twitter Messages\\src\\semeval_twitter_data.txt";

	public static void main (String[] args) {

		//createARFF(FILE);
		
		try {
			runSentimentAnalysis();
		} catch (IOException e) {
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

	public static void runSentimentAnalysis() throws IOException {
		
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("semeval_twitter_data.arff"));
		
		Instances train = new Instances (breader);
		train.setClassIndex(train.numAttributes() - 1);
		
		breader.close();
		
		NaiveBayes nB = new NaiveBayes();
		try {
			nB.buildClassifier(train);
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
			eval.crossValidateModel(nB, train, 10, new Random(1));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println(eval.toSummaryString("\nResults\n=======\n", true));
		System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));		
	}
	

}