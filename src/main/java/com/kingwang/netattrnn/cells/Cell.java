/**   
 * @package	com.kingwang.ctsrnn.lstm
 * @File		RNNCell.java
 * @Crtdate	Jul 3, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.cells;

import java.util.Map;

import org.jblas.DoubleMatrix;

import com.kingwang.netattrnn.batchderv.BatchDerivative;

/**
 *
 * @author King Wang
 * 
 * Jul 3, 2016 10:25:29 PM
 * @version 1.0
 */
public interface Cell {
	
//	public DoubleMatrix yDecode(DoubleMatrix ht);
	
//	public DoubleMatrix dDecode(DoubleMatrix ht);
	
	/**
	 * activation function
	 * 
	 * @param t
	 * @param input
	 * @param node must be the index of current node
	 * @param acts
	 */
	public void active(int t, Map<String, DoubleMatrix> acts, double... params);

	/**
	 * back-propagation through time
	 * 
	 * @param input
	 * @param ndList must be the list of predicting nodes
	 * @param tmList must be the list of time gap between current node and correpsonding predicting node
	 * @param acts
	 * @param lastT
	 */
	public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell);
	
	public void updateParametersByAdaGrad(BatchDerivative batchDerv, double lr);
	
	public void updateParametersByAdam(BatchDerivative batchDerv, double lr
			, double beta1, double beta2, int epochT);
	
	public void writeCellParameter(String outFile, boolean isAttached);
	
	public void loadCellParameter(String cellParamFile);
}
