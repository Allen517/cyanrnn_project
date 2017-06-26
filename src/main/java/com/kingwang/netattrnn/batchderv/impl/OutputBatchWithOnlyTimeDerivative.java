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
import com.kingwang.netattrnn.cons.AlgConsHSoftmax;

/**
 *
 * @author King Wang
 * 
 * Oct 2, 2016 8:29:49 PM
 * @version 1.0
 */
public class OutputBatchWithOnlyTimeDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6945194770004264043L;

//	public DoubleMatrix[] dWxy;
//	public DoubleMatrix[] dWdy;
//	public DoubleMatrix[] dWty;
//	public DoubleMatrix[] dWsy;
//	public DoubleMatrix[] dby;
//	
//	public DoubleMatrix dWxc;
//	public DoubleMatrix dWdc;
//	public DoubleMatrix dWtc;
//	public DoubleMatrix dWsc;
//	public DoubleMatrix dbc;
	
	public DoubleMatrix dWxd;
	public DoubleMatrix dWdd;
	public DoubleMatrix dWsd;
	public DoubleMatrix dWtd;
	public DoubleMatrix dbd;
	
	public DoubleMatrix dWd;
	
	public void clearBatchDerv() {
//		dWxy = null;
//		dWdy = null;
//		dWty = null;
//		dWsy = null;
//		dby = null;
//		
//		dWxc = null;
//		dWdc = null;
//		dWtc = null;
//		dWsc = null;
//		dbc = null;
		
		dWxd = null;
		dWdd = null;
		dWsd = null;
		dWtd = null;
		dbd = null;
		
		dWd = null;
	}

	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		DoubleMatrix[] _dWxy = new DoubleMatrix[AlgConsHSoftmax.cNum];
		DoubleMatrix[] _dWdy = new DoubleMatrix[AlgConsHSoftmax.cNum];
		DoubleMatrix[] _dWty = new DoubleMatrix[AlgConsHSoftmax.cNum];
		DoubleMatrix[] _dWsy = new DoubleMatrix[AlgConsHSoftmax.cNum];
		DoubleMatrix[] _dby = new DoubleMatrix[AlgConsHSoftmax.cNum];
		
		DoubleMatrix _dWxc = acts.get("dWxc");
		DoubleMatrix _dWdc = acts.get("dWdc");
		DoubleMatrix _dWtc = acts.get("dWtc");
		DoubleMatrix _dWsc = acts.get("dWsc");
		DoubleMatrix _dbc = acts.get("dbc");
		
		DoubleMatrix _dWxd = acts.get("dWxd");
		DoubleMatrix _dWdd = acts.get("dWdd");
		DoubleMatrix _dWsd = acts.get("dWsd");
		DoubleMatrix _dWtd = acts.get("dWtd");
		DoubleMatrix _dbd = acts.get("dbd");
		
		DoubleMatrix _dWd = acts.get("dWd");
		
//		if(dWxy==null || dWdy==null || dWty==null || dWsy==null || dby==null
//				|| dWxc==null || dWdc==null || dWtc==null || dWsc==null || dbc==null
//				|| dWxd==null || dWdd==null || dWsd==null || dWtd==null || dbd==null || dWd==null) {
		if(dWxd==null || dWdd==null || dWsd==null || dWtd==null || dbd==null || dWd==null) {
//			dWxy = new DoubleMatrix[AlgConsHSoftmax.cNum];
//			dWdy = new DoubleMatrix[AlgConsHSoftmax.cNum];
//			dWty = new DoubleMatrix[AlgConsHSoftmax.cNum];
//			dWsy = new DoubleMatrix[AlgConsHSoftmax.cNum];
//			dby = new DoubleMatrix[AlgConsHSoftmax.cNum];
//			
//			dWxc = new DoubleMatrix(_dWxc.rows, _dWxc.columns);
//			dWdc = new DoubleMatrix(_dWdc.rows, _dWdc.columns);
//			dWtc = new DoubleMatrix(_dWtc.rows, _dWtc.columns);
//			dWsc = new DoubleMatrix(_dWsc.rows, _dWsc.columns);
//			dbc = new DoubleMatrix(_dbc.rows, _dbc.columns);
			
			dWxd = new DoubleMatrix(_dWxd.rows, _dWxd.columns);
			dWdd = new DoubleMatrix(_dWdd.rows, _dWdd.columns);
			dWsd = new DoubleMatrix(_dWsd.rows, _dWsd.columns);
			dWtd = new DoubleMatrix(_dWtd.rows, _dWtd.columns);
			dbd = new DoubleMatrix(_dbd.rows, _dbd.columns);
			
			dWd = new DoubleMatrix(_dWd.rows, _dWd.columns);
		}
		
//		for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
//			if(!acts.containsKey("dWxy"+c) || !acts.containsKey("dWdy"+c) || !acts.containsKey("dWty"+c)  
//					|| !acts.containsKey("dWsy"+c) || !acts.containsKey("dby"+c)) {
//				continue;
//			}
//			_dWxy[c] = acts.get("dWxy"+c);
//			_dWdy[c] = acts.get("dWdy"+c);
//			_dWty[c] = acts.get("dWty"+c);
//			_dWsy[c] = acts.get("dWsy"+c);
//			_dby[c] = acts.get("dby"+c);
//			 
//			if(dWxy[c]==null || dWdy[c]==null || dWty[c]==null || dWsy[c]==null || dby[c]==null) {
//				dWxy[c] = new DoubleMatrix(_dWxy[c].rows, _dWxy[c].columns);
//				dWdy[c] = new DoubleMatrix(_dWdy[c].rows, _dWdy[c].columns);
//				dWty[c] = new DoubleMatrix(_dWty[c].rows, _dWty[c].columns);
//				dWsy[c] = new DoubleMatrix(_dWsy[c].rows, _dWsy[c].columns);
//				dby[c] = new DoubleMatrix(_dby[c].rows, _dby[c].columns);
//			}
//			 
//			dWxy[c] = dWxy[c].add(_dWxy[c]).mul(avgFac);
//			dWdy[c] = dWdy[c].add(_dWdy[c]).mul(avgFac);
//			dWty[c] = dWty[c].add(_dWty[c]).mul(avgFac);
//			dWsy[c] = dWsy[c].add(_dWsy[c]).mul(avgFac);
//			dby[c] = dby[c].add(_dby[c]).mul(avgFac);
//		}
//		
//		dWxc = dWxc.add(_dWxc).mul(avgFac);
//		dWdc = dWdc.add(_dWdc).mul(avgFac);
//		dWtc = dWtc.add(_dWtc).mul(avgFac);
//		dWsc = dWsc.add(_dWsc).mul(avgFac);
//		dbc = dbc.add(_dbc).mul(avgFac);
		
		dWxd = dWxd.add(_dWxd).mul(avgFac);
		dWdd = dWdd.add(_dWdd).mul(avgFac);
		dWsd = dWsd.add(_dWsd).mul(avgFac);
		dWtd = dWtd.add(_dWtd).mul(avgFac);
		dbd = dbd.add(_dbd).mul(avgFac);
		
		dWd = dWd.add(_dWd).mul(avgFac);
	}
}
