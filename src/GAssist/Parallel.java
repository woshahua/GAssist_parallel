package GAssist_Parallel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;

import keel.Algorithms.Genetic_Rule_Learning.Globals.LogManager;

public class Parallel {

  static boolean running;
  Rand rn;
  
  Classifier globallyBest = null;
  
  /** Creates a new instance of Parallel */
  public Parallel() {
    rn = new Rand();
    rn.initRand(Parameters.seed + 100);
    ParallelGlobals.setRand(rn);
    
    running = true;
  }

  /**
   *  Executes a number of iterations of GA.
   */
  public void run() {
    Thread[] threads = new Thread[Parameters.parallelParts];
	  
    int subPopulationSize;
    if (Parameters.fixedSubPopSize > 0) {
      subPopulationSize = Parameters.fixedSubPopSize;      
    }
    else {
      subPopulationSize = (int) Math.ceil((double) Parameters.popSize / (double) Parameters.parallelParts);      
    }
    Parameters.subPopulationSize = subPopulationSize;
    
    GA[] ga = new GA[Parameters.parallelParts];
  
    PopulationWrapper.initInstancesEvaluation();
    

    
	  for (int i=0; i < Parameters.parallelParts; i++) {
		  ga[i] = new GA(i);
		  
      ga[i].setStrataToUse(i);
      ga[i].initGA();
	  }
	  
    int next;
    int previous;

    for (int i=0; i < Parameters.parallelParts; i++) {
      next = i + 1;
      previous = i - 1;
      
      if (previous < 0) {
        previous = Parameters.parallelParts - 1;
      }
      
      if (next == Parameters.parallelParts) {
        next = 0;
      }
      
      if (Parameters.parallelParts > 1) {
        ga[i].parallelSettings(ga[next], ga[previous]);
      }
    }
	  
	  Statistics[] results = null;
	  
    try {  
		  for (int i=0; i < Parameters.parallelParts; i++) {
		    threads[i] = new Thread(ga[i]);
      }
		  
      for (int i=0; i < Parameters.parallelParts; i++) {
        threads[i].start();
      }
		  
      for (int i=0; i < Parameters.parallelParts; i++) {
        threads[i].join();
      }

      if (Parameters.saveGlobalStatistics) {
        results = new Statistics[Parameters.parallelParts];
        
        for (int i=0; i < Parameters.parallelParts; i++) {
            results[i] = ga[i].getStatistics();
        }
        
        outputParallelData(results);
      }
  	}
    catch (Exception e) {
      Thread.currentThread().interrupt();
      printException(e);
      return;
    }
    
    Classifier globallyBest = getGlobalBestClassifier(ga);
		GA.outputStatistics(globallyBest, new PerformanceAgent() );
  }
  
  // Save the parallel statistics
  public void outputParallelData(Statistics[] results) {
    String line = "";
    
    line += "# Generation: ErrorRate, Fitness, Rule Count (Alive Rules)";
    LogManager.println("");
    LogManager.println(line);
    
    for (int i=0; i < Parameters.numberOfStatistics; i++) {
      double bestFit = Double.MAX_VALUE;
      int bestResult = -1;
      
      for (int j=0; j < Parameters.parallelParts; j++) {
          if (results[j].bestGlobalFit[i] < bestFit) {
              bestFit = results[j].bestGlobalFit[i];
              bestResult = j;
          }
      }
      
      line = "";
      
      line += "Generation " + String.valueOf(i * Parameters.saveStatisticsEveryXGeneration) + ": ";
      line += String.valueOf(100.0 - (100.0 * results[bestResult].bestGlobalAccuracy[i]));
      line += ", ";
      line += String.valueOf((int) results[bestResult].bestGlobalFit[i]);
      line += ", ";
      line += String.valueOf((int) results[bestResult].bestGlobalRules[i]);
      line += " (";
      line += String.valueOf((int) results[bestResult].bestGlobalAliveRules[i]);
      line += ")";
      
      LogManager.println(line);
    }
  }
  
  private Classifier getGlobalBestClassifier(GA[] ga) {
    Classifier inv, bestInv = null;
    
    for (int i=0; i < Parameters.parallelParts; i++) {
      inv = ga[i].getGloballyBestIndividual();
      if (inv.globalCompareToIndividual(bestInv)) {
        bestInv = inv;
      }
    }
    
    return bestInv;
  }
	
	public static void printException(Exception e) {
	      System.err.println("1");
	      System.err.println(e);
	      System.err.println("\n2");
	      System.err.println(e.getMessage());
	      System.err.println("\n3");
	      System.err.println(e.getLocalizedMessage());
	      System.err.println("\n4");
	      System.err.println(e.getCause());
	      System.err.println("\n5");
	      System.err.println(Arrays.toString(e.getStackTrace()));
	      System.err.println("\n6");
	      e.printStackTrace();
	}

}

