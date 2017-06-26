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
public class AttBatchDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6945194770004264043L;

	public DoubleMatrix dV;
	public DoubleMatrix dU;
	public DoubleMatrix dW;
	public DoubleMatrix dbs;
	
	public void clearBatchDerv() {
		dV = null;
		dU = null;
		dW = null;
		dbs = null;
	}

	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		DoubleMatrix _dV = acts.get("dV");
		DoubleMatrix _dU = acts.get("dU");
		DoubleMatrix _dW = acts.get("dW");
		DoubleMatrix _dbs = acts.get("dbs");
		
		if(dV==null) {
			dV = new DoubleMatrix(_dV.rows, _dV.columns);
		}
		if(dU==null) {
			dU = new DoubleMatrix(_dU.rows, _dU.columns);
		}
		if(dW==null) {
			dW = new DoubleMatrix(_dW.rows, _dW.columns);
		}
		if(dbs==null) {
			dbs = new DoubleMatrix(_dbs.rows, _dbs.columns);
		}
		
		dV = dV.add(_dV).mul(avgFac);
		dU = dU.add(_dU).mul(avgFac);
		dW = dW.add(_dW).mul(avgFac);
		dbs = dbs.add(_dbs).mul(avgFac);
	}
}
