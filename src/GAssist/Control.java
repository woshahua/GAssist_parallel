/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. S逍｣chez (luciano@uniovi.es)
    J. Alcal�ｽFdez (jalcala@decsai.ugr.es)
    S. Garc蜒�(sglopez@ujaen.es)
    A. Fern逍｣dez (alberto.fernandez@ujaen.es)
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
 * @author Written by Jaume Bacardit (La Salle, Ram詹｢ Llull University - Barcelona) 28/03/2004
 * @author Modified by Xavi Sol�ｽ(La Salle, Ram詹｢ Llull University - Barcelona) 23/12/2008
 * @version 1.1
 * @since JDK1.2
 * </p>
 */

package GAssist;

import keel.Algorithms.Genetic_Rule_Learning.Globals.*;

public class Control {

  /** Creates a new instance of Control */
  public Control() {
  }

  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    long t1 = System.currentTimeMillis();

    ParserParameters.doParse(args[0]);
    //Parameters.trainFile = args[1];
    //Parameters.testFile = args[2];
    LogManager.initLogManager();
    System.out.println("Test1");
    Rand.initRand();

    GA ga = new GA();
    ga.initGA();
    ga.run();

    LogManager.println(Chronometer.getChrons());
    long t2 = System.currentTimeMillis();
    LogManager.println("Total time: " + ( (t2 - t1) / 1000.0));

    LogManager.closeLog();
  }

}

