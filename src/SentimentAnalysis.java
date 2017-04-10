import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Random;
import java.util.StringTokenizer;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class SentimentAnalysis {

	private static final String FILE = "src/semeval_twitter_data.txt";

	public static void main (String[] args) {

		createARFF(FILE);
		
		try {
			runSentimentAnalysis();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void createARFF(String name) {
		try {
			File file = new File(name);
			File fout = new File("out.txt");
			FileOutputStream fos = new FileOutputStream(fout);
			FileReader fileReader = new FileReader(file);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
			String line;
			
			bw.write("@relation opinion\n@attribute sentence string\n@attribute category {positive,negative,neutral,objective}\n@data\n");
			
			while ((line = bufferedReader.readLine()) != null) {
				bw.write(parseTweet(line));
				bw.newLine();
			}
			fileReader.close();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static String parseTweet(String content) {
		String[] result = content.split("\\t");
		return "' " + result[3].replace("'", "") + " '," + result[2].substring(1, result[2].length() - 1);
  
	}

	public static void runSentimentAnalysis() throws Exception {
		ArffLoader loader = new ArffLoader();
	    loader.setFile(new File("out.txt"));
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