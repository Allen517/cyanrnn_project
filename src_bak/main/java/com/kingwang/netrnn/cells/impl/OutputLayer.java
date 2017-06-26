/**   
 * @package	com.kingwang.cdmrnn.rnn
 * @File		OutputLayer.java
 * @Crtdate	Sep 28, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn.cells.impl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.netrnn.batchderv.BatchDerivative;
import com.kingwang.netrnn.batchderv.impl.OutputBatchDerivative;
import com.kingwang.netrnn.cells.Cell;
import com.kingwang.netrnn.cells.Operator;
import com.kingwang.netrnn.comm.utils.FileUtil;
import com.kingwang.netrnn.cons.AlgCons;
import com.kingwang.netrnn.utils.Activer;
import com.kingwang.netrnn.utils.LoadTypes;
import com.kingwang.netrnn.utils.MatIniter;
import com.kingwang.netrnn.utils.MatIniter.Type;

/**
 *
 * @author King Wang
 * 
 * Sep 28, 2016 5:00:51 PM
 * @version 1.0
 */
public class OutputLayer extends Operator implements Cell, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8868938450690252135L;
	
	public DoubleMatrix Wsy;
    public DoubleMatrix by;
	
	public DoubleMatrix hdWsy;
    public DoubleMatrix hdby;
	
	public DoubleMatrix hd2Wsy;
    public DoubleMatrix hd2by;
    
    public OutputLayer(int outSize, int nodeSize, MatIniter initer) {
        if (initer.getType() == Type.Uniform) {
            this.Wsy = initer.uniform(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
            this.Wsy = initer.gaussian(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
            this.Wsy = initer.svd(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        }
        
        this.hdWsy = new DoubleMatrix(outSize, nodeSize);
        this.hdby = new DoubleMatrix(1, nodeSize);
        
        this.hd2Wsy = new DoubleMatrix(outSize, nodeSize);
        this.hd2by = new DoubleMatrix(1, nodeSize);
    }
    
    public void active(int t, Map<String, DoubleMatrix> acts, double... params) {
    	DoubleMatrix s = acts.get("s"+t);
    	DoubleMatrix hatYt = s.mmul(Wsy).add(by);
        DoubleMatrix predictYt = Activer.softmax(hatYt);
        acts.put("py" + t, predictYt);
    }
    
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {
    	DoubleMatrix dWsy = new DoubleMatrix(Wsy.rows, Wsy.columns);
    	DoubleMatrix dby = new DoubleMatrix(by.rows, by.columns);
    	
    	for (int t = lastT; t > -1; t--) {
            // delta y
            DoubleMatrix py = acts.get("py" + t);
            DoubleMatrix y = acts.get("y" + t);
            DoubleMatrix deltaY = py.sub(y);
            acts.put("dy" + t, deltaY);

            DoubleMatrix s = acts.get("s" + t).transpose();
            dWsy = dWsy.add(s.mmul(deltaY));
            dby = dby.add(deltaY);
    	}
    	
    	acts.put("dWsy", dWsy);
    	acts.put("dby", dby);
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	OutputBatchDerivative batchDerv = (OutputBatchDerivative) derv;
    	
        hdWsy = hdWsy.add(MatrixFunctions.pow(batchDerv.dWsy, 2.));
        hdby = hdby.add(MatrixFunctions.pow(batchDerv.dby, 2.));
        
        Wsy = Wsy.sub(batchDerv.dWsy.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWsy).add(eps),-1.).mul(lr)));
        by = by.sub(batchDerv.dby.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdby).add(eps),-1.).mul(lr)));
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
    						, double beta1, double beta2, int epochT) {
    	
    	OutputBatchDerivative batchDerv = (OutputBatchDerivative) derv;
    	
		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		hd2Wsy = hd2Wsy.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWsy, 2.).mul(1 - beta2));
		hd2by = hd2by.mul(beta2).add(MatrixFunctions.pow(batchDerv.dby, 2.).mul(1 - beta2));
		
		hdWsy = hdWsy.mul(beta1).add(batchDerv.dWsy.mul(1 - beta1));
		hdby = hdby.mul(beta1).add(batchDerv.dby.mul(1 - beta1));

		Wsy = Wsy.sub(
				hdWsy.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wsy.mul(biasBeta2)).add(eps), -1))
				);
		by = by.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2by.mul(biasBeta2)).add(eps), -1.)
				.mul(hdby.mul(biasBeta1)).mul(lr)
				);
    }
    
    public DoubleMatrix yDecode(DoubleMatrix ht) {
		return ht.mmul(Wsy).add(by);
	}

	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#writeCellParameter(java.lang.String, boolean)
	 */
	@Override
	public void writeCellParameter(String outFile, boolean isAttached) {
		OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile, isAttached);
    	FileUtil.writeln(osw, "Wsy");
    	writeMatrix(osw, Wsy);
    	FileUtil.writeln(osw, "by");
    	writeMatrix(osw, by);
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
				if(elems.length<2 && !elems[0].contains(".")) {
					type = LoadTypes.valueOf(elems[0]);
					row = 0;
					continue;
				}
				switch(type) {
					case Wsy: this.Wsy = matrixSetter(row, elems, this.Wsy); break;
					case by: this.by = matrixSetter(row, elems, this.by); break;
				}
				row++;
			}
			
		} catch(IOException e) {
			
		}
	}
}
