/**   
 * @package	com.kingwang.rnncdm.lstm
 * @File		InputNeuron.java
 * @Crtdate	Jun 18, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.cells.impl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.netattrnn.batchderv.BatchDerivative;
import com.kingwang.netattrnn.batchderv.impl.InputBatchDerivative;
import com.kingwang.netattrnn.cells.Cell;
import com.kingwang.netattrnn.cells.Operator;
import com.kingwang.netattrnn.comm.utils.FileUtil;
import com.kingwang.netattrnn.cons.AlgConsHSoftmax;
import com.kingwang.netattrnn.utils.LoadTypes;
import com.kingwang.netattrnn.utils.MatIniter;
import com.kingwang.netattrnn.utils.MatIniter.Type;

/**
 * Data structure for input representation 
 *
 * @author King Wang
 * 
 * Jun 18, 2016 8:04:15 PM
 * @version 1.0
 */
public class InputLayerWithCov extends Operator implements Cell, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1070338506191435106L;
	
	public DoubleMatrix Wx;
	public DoubleMatrix bx;
	
	public DoubleMatrix hdWx;
	public DoubleMatrix hdbx;
	
	public DoubleMatrix hd2Wx;
	public DoubleMatrix hd2bx;
	
    public InputLayerWithCov(int codeSize, int inDynSize, MatIniter initer) {
    	
    	if (initer.getType() == Type.Uniform) {
    		Wx = initer.uniform(codeSize, inDynSize);
    	} else if (initer.getType() == Type.Gaussian) {
    		Wx = initer.gaussian(codeSize, inDynSize);
    	} else if (initer.getType() == Type.SVD) {
    		Wx = initer.svd(codeSize, inDynSize);
    	} else if (initer.getType() == Type.Test) {
    		Wx = DoubleMatrix.zeros(codeSize, inDynSize).add(0.1);;
    	}
    	bx = new DoubleMatrix(1, inDynSize).add(AlgConsHSoftmax.biasInitVal);
    	
    	hdWx = new DoubleMatrix(codeSize, inDynSize);
    	hdbx = new DoubleMatrix(1, inDynSize);
    	hd2Wx = new DoubleMatrix(codeSize, inDynSize);
    	hd2bx = new DoubleMatrix(1, inDynSize);
    }
    
    public InputLayerWithCov(DoubleMatrix Wx) {
    	this.Wx = Wx;
    	bx = new DoubleMatrix(1, Wx.columns).add(AlgConsHSoftmax.biasInitVal);
    	
    	hdWx = new DoubleMatrix(Wx.rows, Wx.columns);
    	hdbx = new DoubleMatrix(bx.rows, bx.columns);
    	hd2Wx = new DoubleMatrix(Wx.rows, Wx.columns);
    	hd2bx = new DoubleMatrix(bx.rows, bx.columns);
    }
    
    public void active(int t, Map<String, DoubleMatrix> acts, double... params) {

    	DoubleMatrix code = acts.get("code"+t);
		DoubleMatrix x = Wx.getRow((int) code.get(0)).add(bx);
		acts.put("x"+t, x);
    }
    
    /**
     * 
     * 
     * @param cell
     * @param nodes must be the input sequence
     * @param acts
     * @param lastT
     */
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {
    	
    	OutputLayerWithCov outLayer = (OutputLayerWithCov) cell[0];
    	AttentionWithCov att = (AttentionWithCov) cell[1];
    	GRUWithCov gru = (GRUWithCov)cell[2];
    	
    	DoubleMatrix dWx = new DoubleMatrix(Wx.rows, Wx.columns);
    	DoubleMatrix dbx = new DoubleMatrix(bx.rows, bx.columns);
    	
    	for (int t = 0; t < lastT + 1; t++) {
    		DoubleMatrix deltaX = null;
    		DoubleMatrix deltaD = acts.get("dd"+t);
    		
    		DoubleMatrix code = acts.get("code"+t);
    		
    		//get cidx
    		DoubleMatrix c = acts.get("cls" + t);
    		int cidx = 0;
    		for(; cidx<c.length; cidx++) {
    			if(c.get(cidx)==1) {
    				break;
    			}
    		}
    		//update input vectors
            if(AlgConsHSoftmax.rnnType.equalsIgnoreCase("gru")) {
            	deltaX = acts.get("dr"+t).mmul(gru.Wxr.transpose())
            					.add(acts.get("dz"+t).mmul(gru.Wxz.transpose()))
            					.add(acts.get("dgh"+t).mmul(gru.Wxh.transpose()));
            }
            deltaX = deltaX.add(acts.get("dy"+t).mmul(outLayer.Wxy[cidx].transpose())
            					.add(acts.get("dCls"+t).mmul(outLayer.Wxc.transpose()))
            					.add(acts.get("dAt"+t).mmul(att.Wxt.transpose())))
            					.add(deltaD.mmul(outLayer.Wxd.transpose()));
    		
            int rowNum = (int) code.get(0);
            dWx.putRow(rowNum, dWx.getRow(rowNum).add(deltaX));
            dbx = dbx.add(deltaX);
        }
    	
    	acts.put("dWx", dWx);
    	acts.put("dbx", dbx);
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	InputBatchDerivative batchDerv = (InputBatchDerivative) derv;
    	
    	hdWx = hdWx.add(MatrixFunctions.pow(batchDerv.dWx, 2.));
    	hdbx = hdbx.add(MatrixFunctions.pow(batchDerv.dbx, 2.));
    	
    	Wx = Wx.sub(batchDerv.dWx.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWx).add(eps), -1).mul(lr)));
    	bx = bx.sub(batchDerv.dbx.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbx).add(eps), -1).mul(lr)));
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
							, double beta1, double beta2, int epochT) {
    
    	double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		InputBatchDerivative batchDerv = (InputBatchDerivative) derv;
		
		hdWx = hdWx.mul(beta1).add(batchDerv.dWx.mul(1 - beta1));
		hd2Wx = hd2Wx.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWx, 2).mul(1 - beta2));
		
		hdbx = hdbx.mul(beta1).add(batchDerv.dbx.mul(1 - beta1));
		hd2bx = hd2bx.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbx, 2).mul(1 - beta2));
		
		Wx = Wx.sub(
				hdWx.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wx.mul(biasBeta2)).add(eps), -1))
				);
    }
    
	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#writeCellParameter(java.lang.String, boolean)
	 */
	@Override
	public void writeCellParameter(String outFile, boolean isAttached) {
		OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile, isAttached);
    	FileUtil.writeln(osw, "Wx");
    	writeMatrix(osw, Wx);
    	FileUtil.writeln(osw, "bx");
    	writeMatrix(osw, bx);
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
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(",");
    			if(elems.length<2) {
    				String typeStr = "Null";
    				String[] typeList = {"Wx", "bx"};
    				for(String tStr : typeList) {
    					if(elems[0].equalsIgnoreCase(tStr)) {
    						typeStr = tStr;
    						break;
    					}
    				}
    				type = LoadTypes.valueOf(typeStr);
    				row = 0;
    				continue;
    			}
    			switch(type) {
	    			case Wx: this.Wx = matrixSetter(row, elems, this.Wx); break;
	    			case bx: this.bx = matrixSetter(row, elems, this.bx); break;
    			}
    			row++;
    		}
    		
    	} catch(IOException e) {
    		
    	}
	}
}
