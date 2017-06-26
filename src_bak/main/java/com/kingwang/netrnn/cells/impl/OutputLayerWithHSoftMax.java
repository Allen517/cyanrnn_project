/**   
 * @package	com.kingwang.rnnmtd.rnn.impl
 * @File		OutputLayerWithHSoftMax.java
 * @Crtdate	Dec 6, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn.cells.impl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.netrnn.batchderv.BatchDerivative;
import com.kingwang.netrnn.batchderv.impl.OutputBatchWithHSoftmaxDerivative;
import com.kingwang.netrnn.cells.Cell;
import com.kingwang.netrnn.cells.Operator;
import com.kingwang.netrnn.comm.utils.FileUtil;
import com.kingwang.netrnn.cons.AlgConsHSoftmax;
import com.kingwang.netrnn.utils.Activer;
import com.kingwang.netrnn.utils.LoadTypes;
import com.kingwang.netrnn.utils.MatIniter;
import com.kingwang.netrnn.utils.MatIniter.Type;

/**
 *
 * @author King Wang
 * 
 * Dec 6, 2016 8:01:19 PM
 * @version 1.0
 */
public class OutputLayerWithHSoftMax extends Operator implements Cell, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8059117236015059106L;
	
	public DoubleMatrix[] Wsy;
    public DoubleMatrix[] by;
    public DoubleMatrix Wsc;
    public DoubleMatrix bsc;
    
	public DoubleMatrix[] hdWsy;
    public DoubleMatrix[] hdby;
    public DoubleMatrix hdWsc;
    public DoubleMatrix hdbsc;
	
	public DoubleMatrix[] hd2Wsy;
    public DoubleMatrix[] hd2by;
    public DoubleMatrix hd2Wsc;
    public DoubleMatrix hd2bsc;
    
    public OutputLayerWithHSoftMax(int outSize, int cNum, MatIniter initer) {
        if (initer.getType() == Type.Uniform) {
            this.Wsc = initer.uniform(outSize, cNum);
            this.bsc = new DoubleMatrix(1, cNum).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
            this.Wsc = initer.gaussian(outSize, cNum);
            this.bsc = new DoubleMatrix(1, cNum).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
            this.Wsc = initer.svd(outSize, cNum);
            this.bsc = new DoubleMatrix(1, cNum).add(AlgConsHSoftmax.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        }
        
        this.hdWsc = new DoubleMatrix(outSize, cNum);
        this.hdbsc = new DoubleMatrix(1, cNum);
        
        this.hd2Wsc = new DoubleMatrix(outSize, cNum);
        this.hd2bsc = new DoubleMatrix(1, cNum);
        
        this.Wsy = new DoubleMatrix[cNum];
		this.by = new DoubleMatrix[cNum];
        this.hdWsy = new DoubleMatrix[cNum];
    	this.hdby = new DoubleMatrix[cNum];
    	this.hd2Wsy = new DoubleMatrix[cNum];
    	this.hd2by = new DoubleMatrix[cNum];
        for(int c=0; c<cNum; c++) {
        	if(AlgConsHSoftmax.nodeSizeInCls[c]<1) {
        		continue;
        	}
        	if (initer.getType() == Type.Uniform) {
        		this.Wsy[c] = initer.uniform(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]).add(AlgConsHSoftmax.biasInitVal);
        	}
        	if (initer.getType() == Type.Gaussian) {
        		this.Wsy[c] = initer.gaussian(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]).add(AlgConsHSoftmax.biasInitVal);
        	}
        	if (initer.getType() == Type.SVD) {
        		this.Wsy[c] = initer.svd(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]).add(AlgConsHSoftmax.biasInitVal);
        	}
        	
        	this.hdWsy[c] = new DoubleMatrix(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hdby[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2Wsy[c] = new DoubleMatrix(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]);
        }
    }
    
    public void active(int t, Map<String, DoubleMatrix> acts, double... params) {
    	
    	int cidx = (int) params[0];
    	
    	DoubleMatrix s = acts.get("s"+t);
    	
    	DoubleMatrix Ct = s.mmul(Wsc).add(bsc);
    	DoubleMatrix predictCt = Activer.softmax(Ct);
    	
    	DoubleMatrix hatYt = s.mmul(Wsy[cidx]).add(by[cidx]);
        DoubleMatrix predictYt = Activer.softmax(hatYt);
        acts.put("py" + t, predictYt);
        acts.put("pc" + t, predictCt);
    }
    
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {

    	DoubleMatrix[] dWsy = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	DoubleMatrix[] dby = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
    		if(Wsy[c]==null || by[c]==null) {
    			continue;
    		}
    		dWsy[c] = new DoubleMatrix(Wsy[c].rows, Wsy[c].columns);
        	dby[c] = new DoubleMatrix(by[c].rows, by[c].columns);
    	}
    	DoubleMatrix dWsc = new DoubleMatrix(Wsc.rows, Wsc.columns);
    	DoubleMatrix dbsc = new DoubleMatrix(bsc.rows, bsc.columns);

    	Set<Integer> histCls = new HashSet<>();
    	for (int t = lastT; t > -1; t--) {
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

            DoubleMatrix s = acts.get("s" + t).transpose();
            dWsy[cidx] = dWsy[cidx].add(s.mmul(deltaY));
            dby[cidx] = dby[cidx].add(deltaY);
            dWsc = dWsc.add(s.mmul(deltaCls));
            dbsc = dbsc.add(deltaCls);
    	}
    	
    	for(int cid : histCls) {
    		acts.put("dWsy" + cid, dWsy[cid]);
    		acts.put("dby" + cid, dby[cid]);
    	}
    	acts.put("dWsc", dWsc);
    	acts.put("dbsc", dbsc);
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	OutputBatchWithHSoftmaxDerivative batchDerv = (OutputBatchWithHSoftmaxDerivative) derv;
    	
    	for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
    		if(hdWsy[c]==null || by[c]==null) {
				continue;
			}
    		hdWsy[c] = hdWsy[c].add(MatrixFunctions.pow(batchDerv.dWsy[c], 2.));
    		hdby[c] = hdby[c].add(MatrixFunctions.pow(batchDerv.dby[c], 2.));
    		
    		Wsy[c] = Wsy[c].sub(batchDerv.dWsy[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWsy[c]).add(eps),-1.).mul(lr)));
    		by[c] = by[c].sub(batchDerv.dby[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdby[c]).add(eps),-1.).mul(lr)));
    	}
    	
    	hdWsc = hdWsc.add(MatrixFunctions.pow(batchDerv.dWsc, 2.));
		hdbsc = hdbsc.add(MatrixFunctions.pow(batchDerv.dbsc, 2.));
		
		Wsc = Wsc.sub(batchDerv.dWsc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWsc).add(eps),-1.).mul(lr)));
		bsc = bsc.sub(batchDerv.dbsc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdbsc).add(eps),-1.).mul(lr)));
        
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
    						, double beta1, double beta2, int epochT) {
    	
    	OutputBatchWithHSoftmaxDerivative batchDerv = (OutputBatchWithHSoftmaxDerivative) derv;
    	
		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
			if(hdWsy[c]==null || hd2Wsy[c]==null || hd2by[c]==null || by[c]==null) {
				continue;
			}
			hd2Wsy[c] = hd2Wsy[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dWsy[c], 2.).mul(1 - beta2));
			hd2by[c] = hd2by[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dby[c], 2.).mul(1 - beta2));
			
			hdWsy[c] = hdWsy[c].mul(beta1).add(batchDerv.dWsy[c].mul(1 - beta1));
			hdby[c] = hdby[c].mul(beta1).add(batchDerv.dby[c].mul(1 - beta1));
			
			Wsy[c] = Wsy[c].sub(
					hdWsy[c].mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wsy[c].mul(biasBeta2)).add(eps), -1))
					);
			by[c] = by[c].sub(
					MatrixFunctions.pow(MatrixFunctions.sqrt(hd2by[c].mul(biasBeta2)).add(eps), -1.)
					.mul(hdby[c].mul(biasBeta1)).mul(lr)
					);
		}
		
		hd2Wsc = hd2Wsc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWsc, 2.).mul(1 - beta2));
		hd2bsc = hd2bsc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbsc, 2.).mul(1 - beta2));
		
		hdWsc = hdWsc.mul(beta1).add(batchDerv.dWsc.mul(1 - beta1));
		hdbsc = hdbsc.mul(beta1).add(batchDerv.dbsc.mul(1 - beta1));
		
		Wsc = Wsc.sub(
				hdWsc.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wsc.mul(biasBeta2)).add(eps), -1))
				);
		bsc = bsc.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bsc.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbsc.mul(biasBeta1)).mul(lr)
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
			if(Wsy[c]==null) {
				continue;
			}
			FileUtil.writeln(osw, "Wsy"+c);
			writeMatrix(osw, Wsy[c]);
			FileUtil.writeln(osw, "by"+c);
			writeMatrix(osw, by[c]);
		}
		FileUtil.writeln(osw, "Wsc");
		writeMatrix(osw, Wsc);
		FileUtil.writeln(osw, "bsc");
		writeMatrix(osw, bsc);
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
				int cidx = -1;
				if(elems.length<2 && !elems[0].contains(".")) {
					String typeStr = "";
					if(elems[0].contains("Wsy")) {
						typeStr = "Wsy";
						cidx = Integer.parseInt(elems[0].substring(3));
					}
					if(elems[0].contains("by")) {
						typeStr = "by";
						cidx = Integer.parseInt(elems[0].substring(2));
					}
					type = LoadTypes.valueOf(typeStr);
					row = 0;
					continue;
				}
				switch(type) {
					case Wsy: this.Wsy[cidx] = matrixSetter(row, elems, this.Wsy[cidx]); break;
					case by: this.by[cidx] = matrixSetter(row, elems, this.by[cidx]); break;
					case Wsc: this.Wsc = matrixSetter(row, elems, this.Wsc); break;
					case bsc: this.bsc = matrixSetter(row, elems, this.bsc); break;
				}
				row++;
			}
		} catch(IOException e) {
			
		}
	}
}
