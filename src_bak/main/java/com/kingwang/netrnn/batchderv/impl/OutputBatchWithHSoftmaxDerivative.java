/**   
 * @package	com.kingwang.cdmrnn.utils
 * @File		AttBatchDerivative.java
 * @Crtdate	Oct 2, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn.batchderv.impl;

import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;

import com.kingwang.netrnn.batchderv.BatchDerivative;
import com.kingwang.netrnn.cons.AlgConsHSoftmax;

/**
 *
 * @author King Wang
 * 
 * Oct 2, 2016 8:29:49 PM
 * @version 1.0
 */
public class OutputBatchWithHSoftmaxDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6945194770004264043L;

	public DoubleMatrix[] dWsy;
	public DoubleMatrix[] dby;
	
	public DoubleMatrix dWsc;
	public DoubleMatrix dbsc;
	
	public void clearBatchDerv() {
		dWsy = null;
		dby = null;
		
		dWsc = null;
		dbsc = null;
	}

	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		DoubleMatrix[] _dWsy = new DoubleMatrix[AlgConsHSoftmax.cNum];
		DoubleMatrix[] _dby = new DoubleMatrix[AlgConsHSoftmax.cNum];
		
		if(dWsy==null || dby==null) {
			dWsy = new DoubleMatrix[AlgConsHSoftmax.cNum];
			dby = new DoubleMatrix[AlgConsHSoftmax.cNum];
		}
		
		for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
			if(!acts.containsKey("dWsy"+c) || !acts.containsKey("dby"+c)) {
				continue;
			}
			_dWsy[c] = acts.get("dWsy"+c);
			_dby[c] = acts.get("dby"+c);
			 
			if(dWsy[c]==null || dby[c]==null) {
				dWsy[c] = new DoubleMatrix(_dWsy[c].rows, _dWsy[c].columns);
				dby[c] = new DoubleMatrix(_dby[c].rows, _dby[c].columns);
			}
			 
			dWsy[c] = dWsy[c].add(_dWsy[c]).mul(avgFac);
			dby[c] = dby[c].add(_dby[c]).mul(avgFac);
		}
		
		DoubleMatrix _dWsc = acts.get("dWsc");
		DoubleMatrix _dbsc = acts.get("dbsc");
		 
		if(dWsc==null || dbsc==null) {
			dWsc = new DoubleMatrix(_dWsc.rows, _dWsc.columns);
			dbsc = new DoubleMatrix(_dbsc.rows, _dbsc.columns);
		}
		 
		dWsc = dWsc.add(_dWsc).mul(avgFac);
		dbsc = dbsc.add(_dbsc).mul(avgFac);
	}
}
