/**   
 * @package	com.kingwang.cdmrnn.utils
 * @File		AttBatchDerivative.java
 * @Crtdate	Oct 2, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.batchderv.impl.hist;

import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;

import com.kingwang.netattrnn.batchderv.BatchDerivative;

/**
 *
 * @author King Wang
 * 
 * Oct 2, 2016 8:29:49 PM
 * @version 1.0
 */
public class CovBatchDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6945194770004264043L;

	public DoubleMatrix dWvv;
	public DoubleMatrix dWav;
	public DoubleMatrix dWhv;
	public DoubleMatrix dWtv;
	public DoubleMatrix dbv;
	
	public void clearBatchDerv() {
		dWvv = null;
		dWav = null;
		dWhv = null;
		dWtv = null;
		dbv = null;
	}

	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		DoubleMatrix _dWvv = acts.get("dWvv");
		DoubleMatrix _dWav = acts.get("dWav");
		DoubleMatrix _dWhv = acts.get("dWhv");
		DoubleMatrix _dWtv = acts.get("dWtv");
		DoubleMatrix _dbv = acts.get("dbv");

		if(dWvv==null || dWav==null || dWhv==null || dWtv==null || dbv==null) {
			dWvv = new DoubleMatrix(_dWvv.rows, _dWvv.columns);
			dWav = new DoubleMatrix(_dWav.rows, _dWvv.columns);
			dWhv = new DoubleMatrix(_dWhv.rows, _dWhv.columns);
			dWtv = new DoubleMatrix(_dWtv.rows, _dWtv.columns);
			dbv = new DoubleMatrix(_dbv.rows, _dbv.columns);
		}
		
		dWvv = dWvv.add(_dWvv).mul(avgFac);
		dWav = dWav.add(_dWav).mul(avgFac);
		dWhv = dWhv.add(_dWhv).mul(avgFac);
		dWtv = dWtv.add(_dWtv).mul(avgFac);
		dbv = dbv.add(_dbv).mul(avgFac);
	}
}
