/**   
 * @package	com.kingwang.rnnmtd.rnn.impl
 * @File		OutputLayerWithHSoftMax.java
 * @Crtdate	Dec 6, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.cells.impl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.netattrnn.batchderv.BatchDerivative;
import com.kingwang.netattrnn.batchderv.impl.OutputBatchWithTimeDerivative;
import com.kingwang.netattrnn.cells.Cell;
import com.kingwang.netattrnn.cells.Operator;
import com.kingwang.netattrnn.comm.utils.FileUtil;
import com.kingwang.netattrnn.cons.AlgConsHSoftmax;
import com.kingwang.netattrnn.utils.Activer;
import com.kingwang.netattrnn.utils.LoadTypes;
import com.kingwang.netattrnn.utils.MatIniter;
import com.kingwang.netattrnn.utils.MatIniter.Type;

/**
 *
 * @author King Wang
 * 
 * Dec 6, 2016 8:01:19 PM
 * @version 1.0
 */
public class OutputLayerWithTime extends Operator implements Cell, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8059117236015059106L;
	
	public DoubleMatrix[] Wxy;
	public DoubleMatrix[] Wdy;
	public DoubleMatrix[] Wsy;
	public DoubleMatrix[] Wty;
    public DoubleMatrix[] by;
    public DoubleMatrix Wxc;
    public DoubleMatrix Wdc;
    public DoubleMatrix Wsc;
    public DoubleMatrix Wtc;
    public DoubleMatrix bc;
    
    public DoubleMatrix[] hdWxy;
    public DoubleMatrix[] hdWdy;
	public DoubleMatrix[] hdWsy;
	public DoubleMatrix[] hdWty;
    public DoubleMatrix[] hdby;
    public DoubleMatrix hdWxc;
    public DoubleMatrix hdWdc;
    public DoubleMatrix hdWsc;
    public DoubleMatrix hdWtc;
    public DoubleMatrix hdbc;
	
    public DoubleMatrix[] hd2Wxy;
    public DoubleMatrix[] hd2Wdy;
	public DoubleMatrix[] hd2Wsy;
	public DoubleMatrix[] hd2Wty;
    public DoubleMatrix[] hd2by;
    public DoubleMatrix hd2Wxc;
    public DoubleMatrix hd2Wdc;
    public DoubleMatrix hd2Wsc;
    public DoubleMatrix hd2Wtc;
    public DoubleMatrix hd2bc;
    
    public DoubleMatrix Wd;
    public DoubleMatrix Wxd;
    public DoubleMatrix Wdd;
    public DoubleMatrix Wsd;
    public DoubleMatrix Wtd;
    public DoubleMatrix bd;
    
    public DoubleMatrix hdWd;
    public DoubleMatrix hdWxd;
    public DoubleMatrix hdWdd;
    public DoubleMatrix hdWsd;
    public DoubleMatrix hdWtd;
    public DoubleMatrix hdbd;
    
    public DoubleMatrix hd2Wd;
    public DoubleMatrix hd2Wxd;
    public DoubleMatrix hd2Wdd;
    public DoubleMatrix hd2Wsd;
    public DoubleMatrix hd2Wtd;
    public DoubleMatrix hd2bd;
    
    public OutputLayerWithTime(int inDynSize, int inFixedSize, int attSize, int hiddenSize
    								, int cNum, MatIniter initer) {
        if (initer.getType() == Type.Uniform) {
        	this.Wxc = initer.uniform(inDynSize, cNum);
        	this.Wdc = initer.uniform(inFixedSize, cNum);
        	this.Wsc = initer.uniform(attSize, cNum);
            this.Wtc = initer.uniform(hiddenSize, cNum);
            this.bc = new DoubleMatrix(1, cNum).add(AlgConsHSoftmax.biasInitVal);
            
            this.Wxd = initer.uniform(inDynSize, 1);
            this.Wdd = initer.uniform(inFixedSize, 1);
            this.Wsd = initer.uniform(attSize, 1);
            this.Wtd = initer.uniform(hiddenSize, 1);
            this.bd = new DoubleMatrix(1).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
        	this.Wxc = initer.gaussian(inDynSize, cNum);
        	this.Wdc = initer.gaussian(inFixedSize, cNum);
        	this.Wsc = initer.gaussian(attSize, cNum);
            this.Wtc = initer.gaussian(hiddenSize, cNum);
            this.bc = new DoubleMatrix(1, cNum).add(AlgConsHSoftmax.biasInitVal);
            
            this.Wxd = initer.gaussian(inDynSize, 1);
            this.Wdd = initer.gaussian(inFixedSize, 1);
            this.Wsd = initer.gaussian(attSize, 1);
            this.Wtd = initer.gaussian(hiddenSize, 1);
            this.bd = new DoubleMatrix(1).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
        	this.Wxc = initer.svd(inDynSize, cNum);
        	this.Wdc = initer.svd(inFixedSize, cNum);
        	this.Wsc = initer.svd(attSize, cNum);
            this.Wtc = initer.svd(hiddenSize, cNum);
            this.bc = new DoubleMatrix(1, cNum).add(AlgConsHSoftmax.biasInitVal);
            
            this.Wxd = initer.svd(inDynSize, 1);
            this.Wdd = initer.svd(inFixedSize, 1);
            this.Wsd = initer.svd(attSize, 1);
            this.Wtd = initer.svd(hiddenSize, 1);
            this.bd = new DoubleMatrix(1).add(AlgConsHSoftmax.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        }
        this.Wd = initer.uniform(1, 1);
        this.hdWd = new DoubleMatrix(1);
        this.hd2Wd = new DoubleMatrix(1);
        
        this.hdWxc = new DoubleMatrix(inDynSize, cNum);
        this.hdWdc = new DoubleMatrix(inFixedSize, cNum);
        this.hdWsc = new DoubleMatrix(attSize, cNum);
        this.hdWtc = new DoubleMatrix(hiddenSize, cNum);
        this.hdbc = new DoubleMatrix(1, cNum);
        
        this.hdWxd = new DoubleMatrix(inDynSize, 1);
        this.hdWdd = new DoubleMatrix(inFixedSize, 1);
        this.hdWsd = new DoubleMatrix(attSize, 1);
        this.hdWtd = new DoubleMatrix(hiddenSize, 1);
        this.hdbd = new DoubleMatrix(1, 1);
        
        this.hd2Wxc = new DoubleMatrix(inDynSize, cNum);
        this.hd2Wdc = new DoubleMatrix(inFixedSize, cNum);
        this.hd2Wsc = new DoubleMatrix(attSize, cNum);
        this.hd2Wtc = new DoubleMatrix(hiddenSize, cNum);
        this.hd2bc = new DoubleMatrix(1, cNum);
        
        this.hd2Wxd = new DoubleMatrix(inDynSize, 1);
        this.hd2Wdd = new DoubleMatrix(inFixedSize, 1);
        this.hd2Wsd = new DoubleMatrix(attSize, 1);
        this.hd2Wtd = new DoubleMatrix(hiddenSize, 1);
        this.hd2bd = new DoubleMatrix(1, 1);
        
        this.Wxy = new DoubleMatrix[cNum];
        this.Wdy = new DoubleMatrix[cNum];
        this.Wsy = new DoubleMatrix[cNum];
        this.Wty = new DoubleMatrix[cNum];
		this.by = new DoubleMatrix[cNum];
		this.hdWxy = new DoubleMatrix[cNum];
		this.hdWdy = new DoubleMatrix[cNum];
		this.hdWsy = new DoubleMatrix[cNum];
        this.hdWty = new DoubleMatrix[cNum];
    	this.hdby = new DoubleMatrix[cNum];
    	this.hd2Wxy = new DoubleMatrix[cNum];
    	this.hd2Wdy = new DoubleMatrix[cNum];
		this.hd2Wsy = new DoubleMatrix[cNum];
    	this.hd2Wty = new DoubleMatrix[cNum];
    	this.hd2by = new DoubleMatrix[cNum];
        for(int c=0; c<cNum; c++) {
        	if(AlgConsHSoftmax.nodeSizeInCls[c]<1) {
        		continue;
        	}
        	if (initer.getType() == Type.Uniform) {
        		this.Wxy[c] = initer.uniform(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wdy[c] = initer.uniform(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wsy[c] = initer.uniform(attSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wty[c] = initer.uniform(hiddenSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]).add(AlgConsHSoftmax.biasInitVal);
        	}
        	if (initer.getType() == Type.Gaussian) {
        		this.Wxy[c] = initer.gaussian(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wdy[c] = initer.gaussian(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wsy[c] = initer.gaussian(attSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wty[c] = initer.gaussian(hiddenSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]).add(AlgConsHSoftmax.biasInitVal);
        	}
        	if (initer.getType() == Type.SVD) {
        		this.Wxy[c] = initer.svd(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wdy[c] = initer.svd(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wsy[c] = initer.svd(attSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wty[c] = initer.svd(hiddenSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]).add(AlgConsHSoftmax.biasInitVal);
        	}
        	
        	this.hdWxy[c] = new DoubleMatrix(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hdWdy[c] = new DoubleMatrix(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hdWsy[c] = new DoubleMatrix(attSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hdWty[c] = new DoubleMatrix(hiddenSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hdby[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2Wxy[c] = new DoubleMatrix(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2Wdy[c] = new DoubleMatrix(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2Wsy[c] = new DoubleMatrix(attSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2Wty[c] = new DoubleMatrix(hiddenSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]);
        }
    }
    
    public void active(int t, Map<String, DoubleMatrix> acts, double... params) {
    	
    	int cidx = (int) params[0];
    	
    	DoubleMatrix x = acts.get("x"+t);
    	DoubleMatrix fixedFeat = acts.get("fixedFeat" + t);
    	DoubleMatrix vecT = acts.get("t"+t);
    	DoubleMatrix s = acts.get("s"+t);
    	
    	DoubleMatrix Ct = x.mmul(Wxc).add(fixedFeat.mmul(Wdc)).add(vecT.mmul(Wtc)).add(s.mmul(Wsc)).add(bc);
    	DoubleMatrix predictCt = Activer.softmax(Ct);
    	
    	DoubleMatrix hatYt = x.mmul(Wxy[cidx]).add(fixedFeat.mmul(Wdy[cidx])).add(vecT.mmul(Wty[cidx]))
    							.add(s.mmul(Wsy[cidx])).add(by[cidx]);
        DoubleMatrix predictYt = Activer.softmax(hatYt);
        acts.put("py" + t, predictYt);
        acts.put("pc" + t, predictCt);

        DoubleMatrix d = x.mmul(Wxd).add(fixedFeat.mmul(Wdd)).add(vecT.mmul(Wtd)).add(s.mmul(Wsd)).add(bd);
        acts.put("decD" + t, d);
    }
    
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {

    	DoubleMatrix[] dWxy = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	DoubleMatrix[] dWdy = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	DoubleMatrix[] dWty = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	DoubleMatrix[] dWsy = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	DoubleMatrix[] dby = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
    		if(Wsy[c]==null || by[c]==null || Wxy[c]==null || Wdy[c]==null || Wty[c]==null) {
    			continue;
    		}
    		dWxy[c] = new DoubleMatrix(Wxy[c].rows, Wxy[c].columns);
    		dWdy[c] = new DoubleMatrix(Wdy[c].rows, Wdy[c].columns);
        	dWty[c] = new DoubleMatrix(Wty[c].rows, Wty[c].columns);
    		dWsy[c] = new DoubleMatrix(Wsy[c].rows, Wsy[c].columns);
        	dby[c] = new DoubleMatrix(by[c].rows, by[c].columns);
    	}
    	DoubleMatrix dWxc = new DoubleMatrix(Wxc.rows, Wxc.columns);
    	DoubleMatrix dWdc = new DoubleMatrix(Wdc.rows, Wdc.columns);
    	DoubleMatrix dWtc = new DoubleMatrix(Wtc.rows, Wtc.columns);
    	DoubleMatrix dWsc = new DoubleMatrix(Wsc.rows, Wsc.columns);
    	DoubleMatrix dbc = new DoubleMatrix(bc.rows, bc.columns);
    	
    	
    	DoubleMatrix dWxd = new DoubleMatrix(Wxd.rows, Wxd.columns);
    	DoubleMatrix dWdd = new DoubleMatrix(Wdd.rows, Wdd.columns);
    	DoubleMatrix dWsd = new DoubleMatrix(Wsd.rows, Wsd.columns);
    	DoubleMatrix dWtd = new DoubleMatrix(Wtd.rows, Wtd.columns);
        DoubleMatrix dbd = new DoubleMatrix(bd.rows, bd.columns);
    	
        DoubleMatrix dWd = new DoubleMatrix(Wd.rows, Wd.columns); 
        
    	Set<Integer> histCls = new HashSet<>();
    	DoubleMatrix tmList = acts.get("tmList");
    	for (int t = lastT; t > -1; t--) {
    		// delta d
    		double tmGap = tmList.get(t);
        	DoubleMatrix decD = acts.get("decD"+t);
	        DoubleMatrix lambda = MatrixFunctions.exp(decD);
        	DoubleMatrix deltaD = lambda.div(Wd)
        						.mul(MatrixFunctions.exp(Wd.mul(tmGap)).sub(1))
    							.sub(1);
            acts.put("dd" + t, deltaD);
            
    		// delta c
    		DoubleMatrix pc = acts.get("pc" + t);
    		DoubleMatrix c = acts.get("cls" + t);
    		DoubleMatrix deltaCls = pc.sub(c);
    		acts.put("dCls" + t, deltaCls);
    		//get cidx
    		int cidx = 0;
    		for(; cidx<c.length; cidx++) {
    			if(c.get(cidx)==1) {
    				break;
    			}
    		}
    		histCls.add(cidx);
    		
            // delta y
            DoubleMatrix py = acts.get("py" + t);
            DoubleMatrix y = acts.get("y" + t);
            DoubleMatrix deltaY = py.sub(y);
            acts.put("dy" + t, deltaY);

            DoubleMatrix x = acts.get("x" + t).transpose();
            DoubleMatrix fixedFeat = acts.get("fixedFeat" + t).transpose();
            DoubleMatrix vecT = acts.get("t" + t).transpose();
            DoubleMatrix s = acts.get("s" + t).transpose();
            dWxy[cidx] = dWxy[cidx].add(x.mmul(deltaY));
            dWdy[cidx] = dWdy[cidx].add(fixedFeat.mmul(deltaY));
            dWty[cidx] = dWty[cidx].add(vecT.mmul(deltaY));
            dWsy[cidx] = dWsy[cidx].add(s.mmul(deltaY));
            dby[cidx] = dby[cidx].add(deltaY);
            dWxc = dWxc.add(x.mmul(deltaCls));
            dWdc = dWdc.add(fixedFeat.mmul(deltaCls));
            dWtc = dWtc.add(vecT.mmul(deltaCls));
            dWsc = dWsc.add(s.mmul(deltaCls));
            dbc = dbc.add(deltaCls);
            
            //delta Whd & bd
            dWxd = dWxd.add(x.mmul(deltaD));
            dWdd = dWdd.add(fixedFeat.mmul(deltaD));
            dWsd = dWsd.add(s.mmul(deltaD));
            dWtd = dWtd.add(vecT.mmul(deltaD));
            dbd = dbd.add(deltaD);
            
            //delta Wd
            dWd = dWd.add(MatrixFunctions.pow(Wd, -2).mul(lambda.mul(-1))
		            				.mul(MatrixFunctions.exp(Wd.mul(tmGap)).sub(1))
		            		.add(MatrixFunctions.pow(Wd, -1).mul(tmGap).mul(lambda)
		            				.mul(MatrixFunctions.exp(Wd.mul(tmGap))))
		            		.sub(tmGap));
    	}
    	
    	for(int cid : histCls) {
    		acts.put("dWxy" + cid, dWxy[cid]);
    		acts.put("dWdy" + cid, dWdy[cid]);
    		acts.put("dWty" + cid, dWty[cid]);
    		acts.put("dWsy" + cid, dWsy[cid]);
    		acts.put("dby" + cid, dby[cid]);
    	}
    	acts.put("dWxc", dWxc);
    	acts.put("dWdc", dWdc);
    	acts.put("dWtc", dWtc);
    	acts.put("dWsc", dWsc);
    	acts.put("dbc", dbc);
    	
    	acts.put("dWxd", dWxd);
    	acts.put("dWdd", dWdd);
    	acts.put("dWsd", dWsd);
    	acts.put("dWtd", dWtd);
        acts.put("dbd", dbd);
        
        acts.put("dWd", dWd);
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	OutputBatchWithTimeDerivative batchDerv = (OutputBatchWithTimeDerivative) derv;
    	
    	for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
    		if(hdWsy[c]==null || by[c]==null || hdWdy[c]==null || hdWxy[c]==null || hdWty[c]==null) {
				continue;
			}
    		if(batchDerv.dWxy[c]==null || batchDerv.dWdy[c]==null || batchDerv.dWty[c]==null 
    				|| batchDerv.dWsy[c]==null || batchDerv.dby[c]==null) {
    			continue;
    		}
    		hdWxy[c] = hdWxy[c].add(MatrixFunctions.pow(batchDerv.dWxy[c], 2.));
    		hdWdy[c] = hdWdy[c].add(MatrixFunctions.pow(batchDerv.dWdy[c], 2.));
    		hdWty[c] = hdWty[c].add(MatrixFunctions.pow(batchDerv.dWty[c], 2.));
    		hdWsy[c] = hdWsy[c].add(MatrixFunctions.pow(batchDerv.dWsy[c], 2.));
    		hdby[c] = hdby[c].add(MatrixFunctions.pow(batchDerv.dby[c], 2.));
    		
    		Wxy[c] = Wxy[c].sub(batchDerv.dWxy[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxy[c]).add(eps),-1.).mul(lr)));
    		Wdy[c] = Wdy[c].sub(batchDerv.dWdy[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdy[c]).add(eps),-1.).mul(lr)));
    		Wty[c] = Wty[c].sub(batchDerv.dWty[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWty[c]).add(eps),-1.).mul(lr)));
    		Wsy[c] = Wsy[c].sub(batchDerv.dWsy[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWsy[c]).add(eps),-1.).mul(lr)));
    		by[c] = by[c].sub(batchDerv.dby[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdby[c]).add(eps),-1.).mul(lr)));
    	}
    	
    	hdWxc = hdWxc.add(MatrixFunctions.pow(batchDerv.dWxc, 2.));
    	hdWdc = hdWdc.add(MatrixFunctions.pow(batchDerv.dWdc, 2.));
    	hdWtc = hdWtc.add(MatrixFunctions.pow(batchDerv.dWtc, 2.));
    	hdWsc = hdWsc.add(MatrixFunctions.pow(batchDerv.dWsc, 2.));
		hdbc = hdbc.add(MatrixFunctions.pow(batchDerv.dbc, 2.));
		
		hdWxd = hdWxd.add(MatrixFunctions.pow(batchDerv.dWxd, 2.));
		hdWdd = hdWdd.add(MatrixFunctions.pow(batchDerv.dWdd, 2.));
		hdWsd = hdWsd.add(MatrixFunctions.pow(batchDerv.dWsd, 2.));
		hdWtd = hdWtd.add(MatrixFunctions.pow(batchDerv.dWtd, 2.));
        hdbd = hdbd.add(MatrixFunctions.pow(batchDerv.dbd, 2.));
        
        hdWd = hdWd.add(MatrixFunctions.pow(batchDerv.dWd, 2.));
		
		Wxc = Wxc.sub(batchDerv.dWxc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxc).add(eps),-1.).mul(lr)));
		Wdc = Wdc.sub(batchDerv.dWdc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdc).add(eps),-1.).mul(lr)));
		Wtc = Wtc.sub(batchDerv.dWtc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWtc).add(eps),-1.).mul(lr)));
		Wsc = Wsc.sub(batchDerv.dWsc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWsc).add(eps),-1.).mul(lr)));
		bc = bc.sub(batchDerv.dbc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdbc).add(eps),-1.).mul(lr)));
		
		Wxd = Wxd.sub(batchDerv.dWxd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxd).add(eps),-1.).mul(lr)));
		Wdd = Wdd.sub(batchDerv.dWdd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdd).add(eps),-1.).mul(lr)));
		Wsd = Wsd.sub(batchDerv.dWsd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWsd).add(eps),-1.).mul(lr)));
		Wtd = Wtd.sub(batchDerv.dWtd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWtd).add(eps),-1.).mul(lr)));
        bd = bd.sub(batchDerv.dbd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbd).add(eps),-1.).mul(lr)));
        
        Wd = Wd.sub(batchDerv.dWd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWd).add(eps),-1.).mul(lr)));
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
    						, double beta1, double beta2, int epochT) {
    	
    	OutputBatchWithTimeDerivative batchDerv = (OutputBatchWithTimeDerivative) derv;
    	
		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
			if(hdWsy[c]==null || hd2Wsy[c]==null || hd2by[c]==null || by[c]==null
					|| hd2Wxy[c]==null || hd2Wty[c]==null) {
				continue;
			}
			if(batchDerv.dWxy[c]==null || batchDerv.dWdy[c]==null || batchDerv.dWty[c]==null 
    				|| batchDerv.dWsy[c]==null || batchDerv.dby[c]==null) {
    			continue;
    		}
			hd2Wxy[c] = hd2Wxy[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxy[c], 2.).mul(1 - beta2));
			hd2Wdy[c] = hd2Wdy[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdy[c], 2.).mul(1 - beta2));
			hd2Wty[c] = hd2Wty[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dWty[c], 2.).mul(1 - beta2));
			hd2Wsy[c] = hd2Wsy[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dWsy[c], 2.).mul(1 - beta2));
			hd2by[c] = hd2by[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dby[c], 2.).mul(1 - beta2));
			
			hdWxy[c] = hdWxy[c].mul(beta1).add(batchDerv.dWxy[c].mul(1 - beta1));
			hdWdy[c] = hdWdy[c].mul(beta1).add(batchDerv.dWdy[c].mul(1 - beta1));
			hdWty[c] = hdWty[c].mul(beta1).add(batchDerv.dWty[c].mul(1 - beta1));
			hdWsy[c] = hdWsy[c].mul(beta1).add(batchDerv.dWsy[c].mul(1 - beta1));
			hdby[c] = hdby[c].mul(beta1).add(batchDerv.dby[c].mul(1 - beta1));
			
			Wxy[c] = Wxy[c].sub(
					hdWxy[c].mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxy[c].mul(biasBeta2)).add(eps), -1))
					);
			Wdy[c] = Wdy[c].sub(
					hdWdy[c].mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdy[c].mul(biasBeta2)).add(eps), -1))
					);
			Wty[c] = Wty[c].sub(
					hdWty[c].mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wty[c].mul(biasBeta2)).add(eps), -1))
					);
			Wsy[c] = Wsy[c].sub(
					hdWsy[c].mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wsy[c].mul(biasBeta2)).add(eps), -1))
					);
			by[c] = by[c].sub(
					MatrixFunctions.pow(MatrixFunctions.sqrt(hd2by[c].mul(biasBeta2)).add(eps), -1.)
					.mul(hdby[c].mul(biasBeta1)).mul(lr)
					);
		}
		
		hd2Wxc = hd2Wxc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxc, 2.).mul(1 - beta2));
		hd2Wdc = hd2Wdc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdc, 2.).mul(1 - beta2));
		hd2Wtc = hd2Wtc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWtc, 2.).mul(1 - beta2));
		hd2Wsc = hd2Wsc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWsc, 2.).mul(1 - beta2));
		hd2bc = hd2bc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbc, 2.).mul(1 - beta2));
		
		hd2Wxd = hd2Wxd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxd, 2.).mul(1 - beta2));
		hd2Wdd = hd2Wdd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdd, 2.).mul(1 - beta2));
		hd2Wsd = hd2Wsd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWsd, 2.).mul(1 - beta2));
		hd2Wtd = hd2Wtd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWtd, 2.).mul(1 - beta2));
		hd2bd = hd2bd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbd, 2.).mul(1 - beta2));
		hd2Wd = hd2Wd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWd, 2.).mul(1 - beta2));
		
		hdWxc = hdWxc.mul(beta1).add(batchDerv.dWxc.mul(1 - beta1));
		hdWdc = hdWdc.mul(beta1).add(batchDerv.dWdc.mul(1 - beta1));
		hdWtc = hdWtc.mul(beta1).add(batchDerv.dWtc.mul(1 - beta1));
		hdWsc = hdWsc.mul(beta1).add(batchDerv.dWsc.mul(1 - beta1));
		hdbc = hdbc.mul(beta1).add(batchDerv.dbc.mul(1 - beta1));
		
		hdWxd = hdWxd.mul(beta1).add(batchDerv.dWxd.mul(1 - beta1));
		hdWdd = hdWdd.mul(beta1).add(batchDerv.dWdd.mul(1 - beta1));
		hdWsd = hdWsd.mul(beta1).add(batchDerv.dWsd.mul(1 - beta1));
		hdWtd = hdWtd.mul(beta1).add(batchDerv.dWtd.mul(1 - beta1));
		hdbd = hdbd.mul(beta1).add(batchDerv.dbd.mul(1 - beta1));
		hdWd = hdWd.mul(beta1).add(batchDerv.dWd.mul(1 - beta1));
		
		Wxc = Wxc.sub(
				hdWxc.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxc.mul(biasBeta2)).add(eps), -1))
				);
		Wdc = Wdc.sub(
				hdWdc.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdc.mul(biasBeta2)).add(eps), -1))
				);
		Wtc = Wtc.sub(
				hdWtc.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wtc.mul(biasBeta2)).add(eps), -1))
				);
		Wsc = Wsc.sub(
				hdWsc.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wsc.mul(biasBeta2)).add(eps), -1))
				);
		bc = bc.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bc.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbc.mul(biasBeta1)).mul(lr)
				);
		
		Wxd = Wxd.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxd.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWxd.mul(biasBeta1)).mul(lr)
				);
		Wdd = Wdd.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdd.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWdd.mul(biasBeta1)).mul(lr)
				);
		Wsd = Wsd.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wsd.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWsd.mul(biasBeta1)).mul(lr)
				);
		Wtd = Wtd.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wtd.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWtd.mul(biasBeta1)).mul(lr)
				);
		bd = bd.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bd.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbd.mul(biasBeta1)).mul(lr)
				);
		
		Wd = Wd.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wd.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWd.mul(biasBeta1)).mul(lr)
				);
    }
    
    public DoubleMatrix yDecode(DoubleMatrix ht, int cidx) {
		return ht.mmul(Wsy[cidx]).add(by[cidx]);
	}

	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#writeCellParameter(java.lang.String, boolean)
	 */
	@Override
	public void writeCellParameter(String outFile, boolean isAttached) {
		
		OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile, isAttached);
		for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
			if(Wsy[c]==null || by[c]==null || Wxy[c]==null || Wty[c]==null) {
				continue;
			}
			FileUtil.writeln(osw, "Wxy"+c);
			writeMatrix(osw, Wxy[c]);
			FileUtil.writeln(osw, "Wdy"+c);
			writeMatrix(osw, Wdy[c]);
			FileUtil.writeln(osw, "Wty"+c);
			writeMatrix(osw, Wty[c]);
			FileUtil.writeln(osw, "Wsy"+c);
			writeMatrix(osw, Wsy[c]);
			FileUtil.writeln(osw, "by"+c);
			writeMatrix(osw, by[c]);
		}
		FileUtil.writeln(osw, "Wxc");
		writeMatrix(osw, Wxc);
		FileUtil.writeln(osw, "Wdc");
		writeMatrix(osw, Wdc);
		FileUtil.writeln(osw, "Wtc");
		writeMatrix(osw, Wtc);
		FileUtil.writeln(osw, "Wsc");
		writeMatrix(osw, Wsc);
		FileUtil.writeln(osw, "bc");
		writeMatrix(osw, bc);
		
		FileUtil.writeln(osw, "Wxd");
		writeMatrix(osw, Wxd);
		FileUtil.writeln(osw, "Wdd");
		writeMatrix(osw, Wdd);
		FileUtil.writeln(osw, "Wsd");
		writeMatrix(osw, Wsd);
		FileUtil.writeln(osw, "Wtd");
		writeMatrix(osw, Wtd);
		FileUtil.writeln(osw, "bd");
		writeMatrix(osw, bd);
		FileUtil.writeln(osw, "Wd");
		writeMatrix(osw, Wd);
	}

	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#loadCellParameter(java.lang.String)
	 */
	@Override
	public void loadCellParameter(String cellParamFile) {
		LoadTypes type = LoadTypes.Null;
		int row = 0;
		
		try(BufferedReader br = FileUtil.getBufferReader(cellParamFile)) {
			String line = null;
			int cidx = -1;
			while((line=br.readLine())!=null) {
				String[] elems = line.split(",");
				if(elems.length<2 && !elems[0].contains(".")) {
					String typeStr = "Null";
					cidx = -1;
					if(elems[0].contains("Wxy")) {
						typeStr = "Wxy";
						cidx = Integer.parseInt(elems[0].substring(3));
					}
					if(elems[0].contains("Wdy")) {
						typeStr = "Wdy";
						cidx = Integer.parseInt(elems[0].substring(3));
					}
					if(elems[0].contains("Wty")) {
						typeStr = "Wty";
						cidx = Integer.parseInt(elems[0].substring(3));
					}
					if(elems[0].contains("Wsy")) {
						typeStr = "Wsy";
						cidx = Integer.parseInt(elems[0].substring(3));
					}
					if(elems[0].contains("by")) {
						typeStr = "by";
						cidx = Integer.parseInt(elems[0].substring(2));
					}
					String[] typeList = {"Wxc", "Wdc", "Wtc", "Wsc", "bsc"
								, "Wxd", "Wdd", "Wsd", "Wtd", "bd", "Wd"};
					for(String tStr : typeList) {
						if(elems[0].contains(tStr)) {
							typeStr = tStr;
							break;
						}
					}
					type = LoadTypes.valueOf(typeStr);
					row = 0;
					continue;
				}
				switch(type) {
					case Wxy: this.Wxy[cidx] = matrixSetter(row, elems, this.Wxy[cidx]); break;
					case Wdy: this.Wdy[cidx] = matrixSetter(row, elems, this.Wdy[cidx]); break;
					case Wty: this.Wty[cidx] = matrixSetter(row, elems, this.Wty[cidx]); break;
					case Wsy: this.Wsy[cidx] = matrixSetter(row, elems, this.Wsy[cidx]); break;
					case by: this.by[cidx] = matrixSetter(row, elems, this.by[cidx]); break;
					case Wxc: this.Wxc = matrixSetter(row, elems, this.Wxc); break;
					case Wdc: this.Wdc = matrixSetter(row, elems, this.Wdc); break;
					case Wtc: this.Wtc = matrixSetter(row, elems, this.Wtc); break;
					case Wsc: this.Wsc = matrixSetter(row, elems, this.Wsc); break;
					case bsc: this.bc = matrixSetter(row, elems, this.bc); break;
					case Wxd: this.Wxd = matrixSetter(row, elems, this.Wxd); break;
					case Wdd: this.Wdd = matrixSetter(row, elems, this.Wdd); break;
					case Wsd: this.Wsd = matrixSetter(row, elems, this.Wsd); break;
					case Wtd: this.Wtd = matrixSetter(row, elems, this.Wtd); break;
					case bd: this.bc = matrixSetter(row, elems, this.bc); break;
					case Wd: this.Wd = matrixSetter(row, elems, this.Wd); break;
				}
				row++;
			}
		} catch(IOException e) {
			
		}
	}
}
