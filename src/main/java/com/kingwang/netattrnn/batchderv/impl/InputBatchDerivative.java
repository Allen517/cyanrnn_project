/**   
 * @package	com.kingwang.cdmrnn.utils
 * @File		InputBatchDerivative.java
 * @Crtdate	Oct 7, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.batchderv.impl;

import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;

import com.kingwang.netattrnn.batchderv.BatchDerivative;

/**
 *
 * @author King Wang
 * 
 * Oct 7, 2016 3:42:08 PM
 * @version 1.0
 */
public class InputBatchDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1590256170028655955L;

	public DoubleMatrix dWx;
	public DoubleMatrix dbx;
	
	public void clearBatchDerv() {
		
		dWx = null;
		dbx = null;
	}
	
	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		
		DoubleMatrix _dWx = acts.get("dWx");
		DoubleMatrix _dbx = acts.get("dbx");
		
		if(dWx==null) {
			dWx = new DoubleMatrix(_dWx.rows, _dWx.columns);
		}
		if(dbx==null) {
			dbx = new DoubleMatrix(_dbx.rows, _dbx.columns);
		}
		
		dWx = dWx.add(_dWx).mul(avgFac);
		dbx = dbx.add(_dbx).mul(avgFac);
	}
}
