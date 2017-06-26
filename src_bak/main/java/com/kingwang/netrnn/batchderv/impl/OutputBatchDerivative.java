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

/**
 *
 * @author King Wang
 * 
 * Oct 2, 2016 8:29:49 PM
 * @version 1.0
 */
public class OutputBatchDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6945194770004264043L;

	public DoubleMatrix dWsy;
	public DoubleMatrix dby;
	
	public void clearBatchDerv() {
		dWsy = null;
		dby = null;
	}

	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		DoubleMatrix _dWsy = acts.get("dWsy");
		DoubleMatrix _dby = acts.get("dby");
		
		if(dWsy==null) {
			dWsy = new DoubleMatrix(_dWsy.rows, _dWsy.columns);
		}
		if(dby==null) {
			dby = new DoubleMatrix(_dby.rows, _dby.columns);
		}
		
		dWsy = dWsy.add(_dWsy).mul(avgFac);
		dby = dby.add(_dby).mul(avgFac);
	}
}
