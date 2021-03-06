/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. S疣chez (luciano@uniovi.es)
    J. Alcal?�Fdez (jalcala@decsai.ugr.es)
    S. Garc�?(sglopez@ujaen.es)
    A. Fern疣dez (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

/**
 * <p>
 * @author Written by Jaume Bacardit (La Salle, Ram��� Llull University - Barcelona) 28/03/2004
 * @author Modified by Xavi Sol?�(La Salle, Ram��� Llull University - Barcelona) 23/12/2008
 * @version 1.1
 * @since JDK1.2
 * </p>
 */


package GAssist_Parallel;

public class ProbabilityManagement {
  public final static int LINEAR = 0;
  public final static int SIGMOIDAL = 1;

  double probStart;
  double probEnd;
  double probLength;
  int evolMode;

  double currentProb;
  double sigmaYLength;
  double sigmaYBase;
  double sigmaXOffset;
  double beta;

  public ProbabilityManagement(double start, double end, int mode) {
    probStart = start;
    probEnd = end;
    evolMode = mode;

    if (mode == LINEAR) {
      probLength = end - start;
      currentProb = start;
    }
    else {
      sigmaYLength = end - start;
      sigmaYBase = start;
      sigmaXOffset = 0.5;
      beta = -10;
    }
  }

  public double incStep() {
    
    int threadNo = ParallelGlobals.getThreadNo();
    
    if (evolMode == LINEAR) {
      currentProb = Parameters.percentageOfLearningPerThread[threadNo]
          * probLength + probStart;
    }
    else {
      currentProb = sigmaYLength
          / (1 + Math.exp(beta
                          * (Parameters.percentageOfLearningPerThread[threadNo] - 0.5)))
          + sigmaYBase;
    }
    return currentProb;
  }
}

