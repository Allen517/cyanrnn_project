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
import com.kingwang.netrnn.batchderv.impl.OutputBatchHasXWithHSoftmaxDerivative;
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
public class OutputLayerHasXWithHSoftMax extends Operator implements Cell, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8059117236015059106L;
	
	public DoubleMatrix[] Wxy;
	public DoubleMatrix[] Wdy;
	public DoubleMatrix[] Wsy;
    public DoubleMatrix[] by;
    public DoubleMatrix Wxc;
    public DoubleMatrix Wdc;
    public DoubleMatrix Wsc;
    public DoubleMatrix bsc;
    
    public DoubleMatrix[] hdWxy;
	public DoubleMatrix[] hdWdy;
	public DoubleMatrix[] hdWsy;
    public DoubleMatrix[] hdby;
    public DoubleMatrix hdWxc;
    public DoubleMatrix hdWdc;
    public DoubleMatrix hdWsc;
    public DoubleMatrix hdbsc;
	
    public DoubleMatrix[] hd2Wxy;
	public DoubleMatrix[] hd2Wdy;
	public DoubleMatrix[] hd2Wsy;
    public DoubleMatrix[] hd2by;
    public DoubleMatrix hd2Wxc;
    public DoubleMatrix hd2Wdc;
    public DoubleMatrix hd2Wsc;
    public DoubleMatrix hd2bsc;
    
    public OutputLayerHasXWithHSoftMax(int inDynSize, int inFixedSize, int outSize, int cNum, MatIniter initer) {
        if (initer.getType() == Type.Uniform) {
        	this.Wxc = initer.uniform(inDynSize, cNum);
        	this.Wdc = initer.uniform(inFixedSize, cNum);
            this.Wsc = initer.uniform(outSize, cNum);
            this.bsc = new DoubleMatrix(1, cNum).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
        	this.Wxc = initer.gaussian(inDynSize, cNum);
        	this.Wdc = initer.gaussian(inFixedSize, cNum);
            this.Wsc = initer.gaussian(outSize, cNum);
            this.bsc = new DoubleMatrix(1, cNum).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
        	this.Wxc = initer.svd(inDynSize, cNum);
        	this.Wdc = initer.svd(inFixedSize, cNum);
            this.Wsc = initer.svd(outSize, cNum);
            this.bsc = new DoubleMatrix(1, cNum).add(AlgConsHSoftmax.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        }
        
        this.hdWxc = new DoubleMatrix(inDynSize, cNum);
    	this.hdWdc = new DoubleMatrix(inFixedSize, cNum);
        this.hdWsc = new DoubleMatrix(outSize, cNum);
        this.hdbsc = new DoubleMatrix(1, cNum);
        
        this.hd2Wxc = new DoubleMatrix(inDynSize, cNum);
    	this.hd2Wdc = new DoubleMatrix(inFixedSize, cNum);
        this.hd2Wsc = new DoubleMatrix(outSize, cNum);
        this.hd2bsc = new DoubleMatrix(1, cNum);
        
        this.Wxy = new DoubleMatrix[cNum];
        this.Wdy = new DoubleMatrix[cNum];
        this.Wsy = new DoubleMatrix[cNum];
		this.by = new DoubleMatrix[cNum];
		this.hdWxy = new DoubleMatrix[cNum];
        this.hdWdy = new DoubleMatrix[cNum];
        this.hdWsy = new DoubleMatrix[cNum];
    	this.hdby = new DoubleMatrix[cNum];
    	this.hd2Wxy = new DoubleMatrix[cNum];
        this.hd2Wdy = new DoubleMatrix[cNum];
    	this.hd2Wsy = new DoubleMatrix[cNum];
    	this.hd2by = new DoubleMatrix[cNum];
        for(int c=0; c<cNum; c++) {
        	if(AlgConsHSoftmax.nodeSizeInCls[c]<1) {
        		continue;
        	}
        	if (initer.getType() == Type.Uniform) {
        		this.Wxy[c] = initer.uniform(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wdy[c] = initer.uniform(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wsy[c] = initer.uniform(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]).add(AlgConsHSoftmax.biasInitVal);
        	}
        	if (initer.getType() == Type.Gaussian) {
        		this.Wxy[c] = initer.gaussian(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wdy[c] = initer.gaussian(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wsy[c] = initer.gaussian(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]).add(AlgConsHSoftmax.biasInitVal);
        	}
        	if (initer.getType() == Type.SVD) {
        		this.Wxy[c] = initer.svd(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wdy[c] = initer.svd(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.Wsy[c] = initer.svd(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        		this.by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]).add(AlgConsHSoftmax.biasInitVal);
        	}
        	
        	this.hdWxy[c] = new DoubleMatrix(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hdWdy[c] = new DoubleMatrix(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hdWsy[c] = new DoubleMatrix(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hdby[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2Wxy[c] = new DoubleMatrix(inDynSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2Wdy[c] = new DoubleMatrix(inFixedSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2Wsy[c] = new DoubleMatrix(outSize, AlgConsHSoftmax.nodeSizeInCls[c]);
        	this.hd2by[c] = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[c]);
        }
    }
    
    public void active(int t, Map<String, DoubleMatrix> acts, double... params) {
    	
    	int cidx = (int) params[0];
    	
    	DoubleMatrix x = acts.get("x"+t);
    	DoubleMatrix fixedFeat = acts.get("fixedFeat"+t);
    	DoubleMatrix s = acts.get("s"+t);
    	
    	DoubleMatrix Ct = x.mmul(Wxc).add(fixedFeat.mmul(Wdc)).add(s.mmul(Wsc)).add(bsc);
    	DoubleMatrix predictCt = Activer.softmax(Ct);
    	
    	DoubleMatrix hatYt = x.mmul(Wxy[cidx]).add(fixedFeat.mmul(Wdy[cidx])).add(s.mmul(Wsy[cidx])).add(by[cidx]);
        DoubleMatrix predictYt = Activer.softmax(hatYt);
        acts.put("py" + t, predictYt);
        acts.put("pc" + t, predictCt);
    }
    
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {

    	DoubleMatrix[] dWxy = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	DoubleMatrix[] dWdy = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	DoubleMatrix[] dWsy = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	DoubleMatrix[] dby = new DoubleMatrix[AlgConsHSoftmax.cNum];
    	for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
    		if(Wsy[c]==null || by[c]==null || Wxy[c]==null || Wdy[c]==null) {
    			continue;
    		}
    		dWxy[c] = new DoubleMatrix(Wxy[c].rows, Wxy[c].columns);
    		dWdy[c] = new DoubleMatrix(Wdy[c].rows, Wdy[c].columns);
    		dWsy[c] = new DoubleMatrix(Wsy[c].rows, Wsy[c].columns);
        	dby[c] = new DoubleMatrix(by[c].rows, by[c].columns);
    	}
    	DoubleMatrix dWxc = new DoubleMatrix(Wxc.rows, Wxc.columns);
    	DoubleMatrix dWdc = new DoubleMatrix(Wdc.rows, Wdc.columns);
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

            DoubleMatrix x = acts.get("x" + t).transpose();
            DoubleMatrix fixedFeat = acts.get("fixedFeat" + t).transpose();
            DoubleMatrix s = acts.get("s" + t).transpose();
            dWxy[cidx] = dWxy[cidx].add(x.mmul(deltaY));
            dWdy[cidx] = dWdy[cidx].add(fixedFeat.mmul(deltaY));
            dWsy[cidx] = dWsy[cidx].add(s.mmul(deltaY));
            dby[cidx] = dby[cidx].add(deltaY);
            dWxc = dWxc.add(x.mmul(deltaCls));
            dWdc = dWdc.add(fixedFeat.mmul(deltaCls));
            dWsc = dWsc.add(s.mmul(deltaCls));
            dbsc = dbsc.add(deltaCls);
    	}
    	
    	for(int cid : histCls) {
    		acts.put("dWxy" + cid, dWxy[cid]);
    		acts.put("dWdy" + cid, dWdy[cid]);
    		acts.put("dWsy" + cid, dWsy[cid]);
    		acts.put("dby" + cid, dby[cid]);
    	}
    	acts.put("dWxc", dWxc);
    	acts.put("dWdc", dWdc);
    	acts.put("dWsc", dWsc);
    	acts.put("dbsc", dbsc);
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	OutputBatchHasXWithHSoftmaxDerivative batchDerv = (OutputBatchHasXWithHSoftmaxDerivative) derv;
    	
    	for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
    		if(hdWxy[c]==null || hdWdy[c]==null || hdWsy[c]==null || by[c]==null) {
				continue;
			}
    		if(batchDerv.dWxy[c]==null || batchDerv.dWdy[c]==null || batchDerv.dWsy[c]==null
					|| batchDerv.dby[c]==null) {
				continue;
			}
    		hdWxy[c] = hdWxy[c].add(MatrixFunctions.pow(batchDerv.dWxy[c], 2.));
    		hdWdy[c] = hdWdy[c].add(MatrixFunctions.pow(batchDerv.dWdy[c], 2.));
    		hdWsy[c] = hdWsy[c].add(MatrixFunctions.pow(batchDerv.dWsy[c], 2.));
    		hdby[c] = hdby[c].add(MatrixFunctions.pow(batchDerv.dby[c], 2.));
    		
    		Wxy[c] = Wxy[c].sub(batchDerv.dWxy[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxy[c]).add(eps),-1.).mul(lr)));
    		Wdy[c] = Wdy[c].sub(batchDerv.dWdy[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdy[c]).add(eps),-1.).mul(lr)));
    		Wsy[c] = Wsy[c].sub(batchDerv.dWsy[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWsy[c]).add(eps),-1.).mul(lr)));
    		by[c] = by[c].sub(batchDerv.dby[c].mul(
    				MatrixFunctions.pow(MatrixFunctions.sqrt(hdby[c]).add(eps),-1.).mul(lr)));
    	}
    	
    	hdWxc = hdWxc.add(MatrixFunctions.pow(batchDerv.dWxc, 2.));
    	hdWdc = hdWdc.add(MatrixFunctions.pow(batchDerv.dWdc, 2.));
    	hdWsc = hdWsc.add(MatrixFunctions.pow(batchDerv.dWsc, 2.));
		hdbsc = hdbsc.add(MatrixFunctions.pow(batchDerv.dbsc, 2.));
		
		Wxc = Wxc.sub(batchDerv.dWxc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxc).add(eps),-1.).mul(lr)));
		Wdc = Wdc.sub(batchDerv.dWdc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdc).add(eps),-1.).mul(lr)));
		Wsc = Wsc.sub(batchDerv.dWsc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWsc).add(eps),-1.).mul(lr)));
		bsc = bsc.sub(batchDerv.dbsc.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdbsc).add(eps),-1.).mul(lr)));
        
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
    						, double beta1, double beta2, int epochT) {
    	
    	OutputBatchHasXWithHSoftmaxDerivative batchDerv = (OutputBatchHasXWithHSoftmaxDerivative) derv;
    	
		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
			if(hdWxy[c]==null || hd2Wxy[c]==null || hdWdy[c]==null || hd2Wdy[c]==null 
					|| hdWsy[c]==null || hd2Wsy[c]==null || hd2by[c]==null || by[c]==null) {
				continue;
			}
			if(batchDerv.dWxy[c]==null || batchDerv.dWdy[c]==null || batchDerv.dWsy[c]==null
					|| batchDerv.dby[c]==null) {
				continue;
			}
			hd2Wxy[c] = hd2Wxy[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxy[c], 2.).mul(1 - beta2));
			hd2Wdy[c] = hd2Wdy[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdy[c], 2.).mul(1 - beta2));
			hd2Wsy[c] = hd2Wsy[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dWsy[c], 2.).mul(1 - beta2));
			hd2by[c] = hd2by[c].mul(beta2).add(MatrixFunctions.pow(batchDerv.dby[c], 2.).mul(1 - beta2));
			
			hdWxy[c] = hdWxy[c].mul(beta1).add(batchDerv.dWxy[c].mul(1 - beta1));
			hdWdy[c] = hdWdy[c].mul(beta1).add(batchDerv.dWdy[c].mul(1 - beta1));
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
		hd2Wsc = hd2Wsc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWsc, 2.).mul(1 - beta2));
		hd2bsc = hd2bsc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbsc, 2.).mul(1 - beta2));
		
		hdWxc = hdWxc.mul(beta1).add(batchDerv.dWxc.mul(1 - beta1));
		hdWdc = hdWdc.mul(beta1).add(batchDerv.dWdc.mul(1 - beta1));
		hdWsc = hdWsc.mul(beta1).add(batchDerv.dWsc.mul(1 - beta1));
		hdbsc = hdbsc.mul(beta1).add(batchDerv.dbsc.mul(1 - beta1));
		
		Wxc = Wxc.sub(
				hdWxc.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxc.mul(biasBeta2)).add(eps), -1))
				);
		Wdc = Wdc.sub(
				hdWdc.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdc.mul(biasBeta2)).add(eps), -1))
				);
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
			if(Wsy[c]==null || by[c]==null || Wxy[c]==null) {
				continue;
			}
			FileUtil.writeln(osw, "Wxy"+c);
			writeMatrix(osw, Wxy[c]);
			FileUtil.writeln(osw, "Wdy"+c);
			writeMatrix(osw, Wdy[c]);
			FileUtil.writeln(osw, "Wsy"+c);
			writeMatrix(osw, Wsy[c]);
			FileUtil.writeln(osw, "by"+c);
			writeMatrix(osw, by[c]);
		}
		FileUtil.writeln(osw, "Wxc");
		writeMatrix(osw, Wxc);
		FileUtil.writeln(osw, "Wdc");
		writeMatrix(osw, Wdc);
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
					if(elems[0].contains("Wsy")) {
						typeStr = "Wsy";
						cidx = Integer.parseInt(elems[0].substring(3));
					}
					if(elems[0].contains("by")) {
						typeStr = "by";
						cidx = Integer.parseInt(elems[0].substring(2));
					}
					String[] typeList = {"Wxc", "Wdc", "Wsc", "bsc"};
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
					case Wsy: this.Wsy[cidx] = matrixSetter(row, elems, this.Wsy[cidx]); break;
					case by: this.by[cidx] = matrixSetter(row, elems, this.by[cidx]); break;
					case Wxc: this.Wxc = matrixSetter(row, elems, this.Wxc); break;
					case Wdc: this.Wdc = matrixSetter(row, elems, this.Wdc); break;
					case Wsc: this.Wsc = matrixSetter(row, elems, this.Wsc); break;
					case bsc: this.bsc = matrixSetter(row, elems, this.bsc); break;
				}
				row++;
			}
		} catch(IOException e) {
			
		}
	}
}
