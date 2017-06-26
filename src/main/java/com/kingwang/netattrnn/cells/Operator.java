/**   
 * @package	com.kingwang.rnncdm.lstm
 * @File		Neuron.java
 * @Crtdate	Jun 18, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.cells;

import java.io.OutputStreamWriter;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.netattrnn.comm.utils.FileUtil;

/**
 *
 * @author King Wang
 * 
 * Jun 18, 2016 9:28:32 PM
 * @version 1.0
 */
public class Operator {

	protected double eps = 10e-8;

	protected DoubleMatrix clip(DoubleMatrix x) {
        return x;
    }
	
	protected DoubleMatrix deriveExp(DoubleMatrix f) {
        return f.mul(DoubleMatrix.ones(1, f.length).sub(f));
    }
    
	protected DoubleMatrix deriveTanh(DoubleMatrix f) {
        return DoubleMatrix.ones(1, f.length).sub(MatrixFunctions.pow(f, 2));
    }
    
	protected void writeMatrix(OutputStreamWriter osw, DoubleMatrix mat) {
    	
    	for(int r=0; r<mat.getRows(); r++) {
    		String wrtLn = "";
    		for(int c=0; c<mat.getColumns(); c++) {
    			wrtLn += Double.toString(mat.get(r, c))+",";
    		}
    		FileUtil.writeln(osw, wrtLn.substring(0, wrtLn.length()-1));
    	}
    }
	
	protected DoubleMatrix matrixSetter(int row, String[] elems, DoubleMatrix x) {
    	
    	int col = x.getColumns();
    	if(elems.length!=col) {
    		System.err.println("Matrix setter in ModelLoader meets problem: the column number in deSizefile" +
    				" is not equal to the number in matrix");
    		return DoubleMatrix.EMPTY;
    	}
    	
    	for(int k=0; k<elems.length; k++) {
    		x.put(row, k, Double.parseDouble(elems[k]));
    	}
    	
    	return x;
    }
}
