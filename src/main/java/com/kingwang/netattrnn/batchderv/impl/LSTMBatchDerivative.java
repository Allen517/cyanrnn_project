/**   
 * @package	com.kingwang.rnncdm.layers
 * @File		InputNeuron.java
 * @Crtdate	May 17, 2016
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
 * May 17, 2016 8:49:16 PM
 * @version 1.0
 */
public class LSTMBatchDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3105102557880200595L;
	
	public DoubleMatrix dWxi;
	public DoubleMatrix dWdi;
	public DoubleMatrix dWhi;
	public DoubleMatrix dWci;
	public DoubleMatrix dbi;
    
	public DoubleMatrix dWxf;
	public DoubleMatrix dWdf;
	public DoubleMatrix dWhf;
	public DoubleMatrix dWcf;
	public DoubleMatrix dbf;
    
	public DoubleMatrix dWxc;
	public DoubleMatrix dWdc;
	public DoubleMatrix dWhc;
	public DoubleMatrix dbc;
    
	public DoubleMatrix dWxo;
	public DoubleMatrix dWdo;
	public DoubleMatrix dWho;
	public DoubleMatrix dWco;
	public DoubleMatrix dbo;
    
	public DoubleMatrix dWhd;
	public DoubleMatrix dbd;
	
	public DoubleMatrix dWhy;
	public DoubleMatrix dby;
	
	public DoubleMatrix dxMat;
	
	public void clearBatchDerv() {
		
		dWxi = null;
		dWdi = null;
		dWhi = null;
		dWci = null;
		dbi = null;
		
		dWxf = null;
		dWdf = null;
		dWhf = null;
		dWcf = null;
		dbf = null;
		
		dWxc = null;
		dWdc = null;
		dWhc = null;
		dbc = null;
		
		dWxo = null;
		dWdo = null;
		dWho = null;
		dWco = null;
		dbo = null;
		
		dWhd = null;
		dbd = null;
		
		dWhy = null;
		dby = null;
		
		dxMat = null;
	}
	
	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		
		DoubleMatrix _dWxi = acts.get("dWxi");
		DoubleMatrix _dWdi = acts.get("dWdi");
		DoubleMatrix _dWhi = acts.get("dWhi");
		DoubleMatrix _dWci = acts.get("dWci");
		DoubleMatrix _dbi = acts.get("dbi");
		
		DoubleMatrix _dWxf = acts.get("dWxf");
		DoubleMatrix _dWdf = acts.get("dWdf");
		DoubleMatrix _dWhf = acts.get("dWhf");
		DoubleMatrix _dWcf = acts.get("dWcf");
		DoubleMatrix _dbf = acts.get("dbf");
		
		DoubleMatrix _dWxc = acts.get("dWxc");
		DoubleMatrix _dWdc = acts.get("dWdc");
		DoubleMatrix _dWhc = acts.get("dWhc");
		DoubleMatrix _dbc = acts.get("dbc");
		
		DoubleMatrix _dWxo = acts.get("dWxo");
		DoubleMatrix _dWdo = acts.get("dWdo");
		DoubleMatrix _dWho = acts.get("dWho");
		DoubleMatrix _dWco = acts.get("dWco");
		DoubleMatrix _dbo = acts.get("dbo");
		
		DoubleMatrix _dWhd = acts.get("dWhd");
		DoubleMatrix _dbd = acts.get("dbd");
		
		DoubleMatrix _dWhy = acts.get("dWhy");
		DoubleMatrix _dby = acts.get("dby");
		
		DoubleMatrix _dxMat = acts.get("dxMat");
		
		if(dWxi==null) {
			dWxi = new DoubleMatrix(_dWxi.rows, _dWxi.columns);
		}
		if(dWdi==null) {
			dWdi = new DoubleMatrix(_dWdi.rows, _dWdi.columns);
		}
		if(dWhi==null) {
			dWhi = new DoubleMatrix(_dWhi.rows, _dWhi.columns);
		}
		if(dWci==null) {
			dWci = new DoubleMatrix(_dWci.rows, _dWci.columns);
		}
		if(dbi==null) {
			dbi = new DoubleMatrix(_dbi.rows, _dbi.columns);
		}
		
		if(dWxf==null) {
			dWxf = new DoubleMatrix(_dWxf.rows, _dWxf.columns);
		}
		if(dWdf==null) {
			dWdf = new DoubleMatrix(_dWdf.rows, _dWdf.columns);
		}
		if(dWhf==null) {
			dWhf = new DoubleMatrix(_dWhf.rows, _dWhf.columns);
		}
		if(dWcf==null) {
			dWcf = new DoubleMatrix(_dWcf.rows, _dWcf.columns);
		}
		if(dbf==null) {
			dbf = new DoubleMatrix(_dbf.rows, _dbf.columns);
		}
		
		if(dWxc==null) {
			dWxc = new DoubleMatrix(_dWxc.rows, _dWxc.columns);
		}
		if(dWdc==null) {
			dWdc = new DoubleMatrix(_dWdc.rows, _dWdc.columns);
		}
		if(dWhc==null) {
			dWhc = new DoubleMatrix(_dWhc.rows, _dWhc.columns);
		}
		if(dbc==null) {
			dbc = new DoubleMatrix(_dbc.rows, _dbc.columns);
		}
		
		if(dWxo==null) {
			dWxo = new DoubleMatrix(_dWxo.rows, _dWxo.columns);
		}
		if(dWdo==null) {
			dWdo = new DoubleMatrix(_dWdo.rows, _dWdo.columns);
		}
		if(dWho==null) {
			dWho = new DoubleMatrix(_dWho.rows, _dWho.columns);
		}
		if(dWco==null) {
			dWco = new DoubleMatrix(_dWco.rows, _dWco.columns);
		}
		if(dbo==null) {
			dbo = new DoubleMatrix(_dbo.rows, _dbo.columns);
		}
		
		if(dWhd==null) {
			dWhd = new DoubleMatrix(_dWhd.rows, _dWhd.columns);
		}
		if(dbd==null) {
			dbd = new DoubleMatrix(_dbd.rows, _dbd.columns);
		}
		
		if(dWhy==null) {
			dWhy = new DoubleMatrix(_dWhy.rows, _dWhy.columns);
		}
		if(dby==null) {
			dby = new DoubleMatrix(_dby.rows, _dby.columns);
		}
		
		if(dxMat==null) {
			dxMat = new DoubleMatrix(_dxMat.rows, _dxMat.columns);
		}
		
		dWxi = dWxi.add(_dWxi).mul(avgFac);
		dWdi = dWdi.add(_dWdi).mul(avgFac);
		dWhi = dWhi.add(_dWhi).mul(avgFac);
		dWci = dWci.add(_dWci).mul(avgFac);
		dbi = dbi.add(_dbi).mul(avgFac);
		
		dWxf = dWxf.add(_dWxf).mul(avgFac);
		dWdf = dWdf.add(_dWdf).mul(avgFac);
		dWhf = dWhf.add(_dWhf).mul(avgFac);
		dWcf = dWcf.add(_dWcf).mul(avgFac);
		dbf = dbf.add(_dbf).mul(avgFac);
		
		dWxc = dWxc.add(_dWxc).mul(avgFac);
		dWdc = dWdc.add(_dWdc).mul(avgFac);
		dWhc = dWhc.add(_dWhc).mul(avgFac);
		dbc = dbc.add(_dbc).mul(avgFac);
		
		dWxo = dWxo.add(_dWxo).mul(avgFac);
		dWdo = dWdo.add(_dWdo).mul(avgFac);
		dWho = dWho.add(_dWho).mul(avgFac);
		dWco = dWco.add(_dWco).mul(avgFac);
		dbo = dbo.add(_dbo).mul(avgFac);
		
		dWhd = dWhd.add(_dWhd).mul(avgFac);
		dbd = dbd.add(_dbd).mul(avgFac);
		
		dWhy = dWhy.add(_dWhy).mul(avgFac);
		dby = dby.add(_dby).mul(avgFac);
		
		dxMat = dxMat.add(_dxMat).mul(avgFac);
	}

	public DoubleMatrix getdxMat() {
		return dxMat;
	}
	
}
