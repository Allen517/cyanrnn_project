/**   
 * @package	com.kingwang.cdmrnn.utils
 * @File		AttBatchDerivative.java
 * @Crtdate	Oct 2, 2016
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
 * Oct 2, 2016 8:29:49 PM
 * @version 1.0
 */
public class AttBatchDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6945194770004264043L;

	public DoubleMatrix dWxt;
	public DoubleMatrix dWdt;
	public DoubleMatrix dWtt;
	public DoubleMatrix dWst;
	public DoubleMatrix dbt;
	
	public DoubleMatrix dV;
	public DoubleMatrix dU;
	public DoubleMatrix dW;
	public DoubleMatrix dbs;
	
	public void clearBatchDerv() {
		dWxt = null;
		dWdt = null;
		dWtt = null;
		dWst = null;
		dbt = null;
		
		dV = null;
		dU = null;
		dW = null;
		dbs = null;
	}

	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		DoubleMatrix _dWxt = acts.get("dWxt");
		DoubleMatrix _dWdt = acts.get("dWdt");
		DoubleMatrix _dWtt = acts.get("dWtt");
		DoubleMatrix _dWst = acts.get("dWst");
		DoubleMatrix _dbt = acts.get("dbt");
		
		DoubleMatrix _dV = acts.get("dV");
		DoubleMatrix _dU = acts.get("dU");
		DoubleMatrix _dW = acts.get("dW");
		DoubleMatrix _dbs = acts.get("dbs");
		
		if(dWxt==null || dWdt==null || dWtt==null || dWst==null || dbt==null) {
			dWxt = new DoubleMatrix(_dWxt.rows, _dWxt.columns);
			dWdt = new DoubleMatrix(_dWdt.rows, _dWdt.columns);
			dWtt = new DoubleMatrix(_dWtt.rows, _dWtt.columns);
			dWst = new DoubleMatrix(_dWst.rows, _dWst.columns);
			dbt = new DoubleMatrix(_dbt.rows, _dbt.columns);
		}
		
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
		
		dWxt = dWxt.add(_dWxt).mul(avgFac);
		dWdt = dWdt.add(_dWdt).mul(avgFac);
		dWtt = dWtt.add(_dWtt).mul(avgFac);
		dWst = dWst.add(_dWst).mul(avgFac);
		dbt = dbt.add(_dbt).mul(avgFac);
		
		dV = dV.add(_dV).mul(avgFac);
		dU = dU.add(_dU).mul(avgFac);
		dW = dW.add(_dW).mul(avgFac);
		dbs = dbs.add(_dbs).mul(avgFac);
	}
}
