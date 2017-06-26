/**   
 * @package	com.kingwang.cdmrnn.rnn
 * @File		Attention.java
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
import com.kingwang.netrnn.batchderv.impl.AttBatchDerivative;
import com.kingwang.netrnn.cells.Cell;
import com.kingwang.netrnn.cells.Operator;
import com.kingwang.netrnn.comm.utils.FileUtil;
import com.kingwang.netrnn.comm.utils.StringHelper;
import com.kingwang.netrnn.cons.AlgConsHSoftmax;
import com.kingwang.netrnn.utils.Activer;
import com.kingwang.netrnn.utils.LoadTypes;
import com.kingwang.netrnn.utils.MatIniter;
import com.kingwang.netrnn.utils.MatIniter.Type;

/**
 *
 * @author King Wang
 * 
 * Sep 28, 2016 3:19:24 PM
 * @version 1.0
 */
public class AttentionHasX extends Operator implements Cell, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3581712325656989072L;
	
	public DoubleMatrix V;
	public DoubleMatrix W;
	public DoubleMatrix U;
	public DoubleMatrix bs;
	
	public DoubleMatrix hdV;
	public DoubleMatrix hdW;
	public DoubleMatrix hdU;
	public DoubleMatrix hdbs;

	public DoubleMatrix hd2V;
	public DoubleMatrix hd2W;
	public DoubleMatrix hd2U;
	public DoubleMatrix hd2bs;
	
	private int outSize = 0;
	
	public AttentionHasX(int outSize, MatIniter initer) {
		hdV = new DoubleMatrix(1, outSize);
		hdW = new DoubleMatrix(outSize, outSize);
		hdU = new DoubleMatrix(outSize, outSize);
		hdbs = new DoubleMatrix(1, outSize);
		
		hd2V = new DoubleMatrix(1, outSize);
		hd2W = new DoubleMatrix(outSize, outSize);
		hd2U = new DoubleMatrix(outSize, outSize);
		hd2bs = new DoubleMatrix(1, outSize);
		
		if (initer.getType() == Type.Uniform) {
			V = initer.uniform(1, outSize);
			W = initer.uniform(outSize, outSize);
			U = initer.uniform(outSize, outSize);
			bs = new DoubleMatrix(1, outSize).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
        	V = initer.gaussian(1, outSize);
    		W = initer.gaussian(outSize, outSize);
    		U = initer.gaussian(outSize, outSize);
    		bs = new DoubleMatrix(1, outSize).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
        	V = initer.svd(1, outSize);
    		W = initer.svd(outSize, outSize);
    		U = initer.svd(outSize, outSize);
    		bs = new DoubleMatrix(1, outSize).add(AlgConsHSoftmax.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        	V = new DoubleMatrix(1, outSize).add(0.1);
    		W = new DoubleMatrix(outSize, outSize).add(0.1);
    		U = new DoubleMatrix(outSize, outSize).add(0.2);
    		bs = new DoubleMatrix(1, outSize).add(0.4);
        }
		
		this.outSize = outSize;
	}
	
	public void active(int t, Map<String, DoubleMatrix> acts, double... params) {
		
		DoubleMatrix prevS = null;
		if(t>0) {
			prevS = acts.get("s"+(t-1));
		} else {
			prevS = DoubleMatrix.zeros(1, outSize);
		}
		int eSize = Math.min(t+1, AlgConsHSoftmax.windowSize);
		DoubleMatrix eWeight = new DoubleMatrix(eSize);
		
		DoubleMatrix gs = new DoubleMatrix(eSize, outSize);
//		DoubleMatrix tanhGs = new DoubleMatrix(t+1, outSize);
		int bsIdx = Math.max(0, t-AlgConsHSoftmax.windowSize+1);
		for(int k=bsIdx; k<t+1; k++) {
			DoubleMatrix hk = acts.get("h"+k);
//			gs.putRow(k, prevS.mmul(paramW).add(hk.mmul(paramU)).add(bs));
			DoubleMatrix gsk = Activer.tanh(prevS.mmul(W).add(hk.mmul(U)).add(bs));
			gs.putRow(k-bsIdx, gsk);
//			double etk = V.mmul(gsk.transpose()).get(0);
			eWeight.putRow(k-bsIdx, V.mmul(gsk.transpose()));
		}
		acts.put("gs"+t, gs);
//		acts.put("tanhGs"+t, tanhGs);
		
		DoubleMatrix alpha = Activer.softmax(eWeight.transpose()).transpose();
		acts.put("alpha"+t, alpha);
		
		DoubleMatrix s = new DoubleMatrix(1, prevS.columns);
		for(int k=bsIdx; k<t+1; k++) {
			DoubleMatrix hk = acts.get("h"+k);
			s = s.add(hk.mul(alpha.get(k-bsIdx)));
		}
		acts.put("s"+t, s);
	}
	
	public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {

//		OutputLayerWithHSoftMax outLayer = (OutputLayerWithHSoftMax)cell[0];
		OutputLayerHasXWithHSoftMax outLayer = (OutputLayerHasXWithHSoftMax)cell[0];
		
		for (int t = lastT; t > -1; t--) {
			DoubleMatrix deltaY = acts.get("dy"+t);
			DoubleMatrix deltaCls = acts.get("dCls"+t);
			//get cidx
			DoubleMatrix c = acts.get("cls" + t);
    		int cidx = 0;
    		for(; cidx<c.length; cidx++) {
    			if(c.get(cidx)==1) {
    				break;
    			}
    		}
    		
            // delta s
			DoubleMatrix deltaS = null;
			if(t==lastT) {
				deltaS = deltaY.mmul(outLayer.Wsy[cidx].transpose()).add(deltaCls.mmul(outLayer.Wsc.transpose()));
			} else {
				deltaS = deltaY.mmul(outLayer.Wsy[cidx].transpose()).add(deltaCls.mmul(outLayer.Wsc.transpose()));
				DoubleMatrix lateDgs = acts.get("dgs"+(t+1));
				int lateESize = Math.min(t+2, AlgConsHSoftmax.windowSize);
				deltaS = deltaS.add(DoubleMatrix.ones(1, lateESize).mmul(lateDgs.mmul(W.transpose())));
			}
			acts.put("ds"+t, deltaS);
			// delta alpha
			int bsIdx = Math.max(0, t-AlgConsHSoftmax.windowSize+1);
			int eSize = Math.min(t+1, AlgConsHSoftmax.windowSize);
			DoubleMatrix deltaAlpha = new DoubleMatrix(eSize);
			for(int j=bsIdx; j<t+1; j++) {
				DoubleMatrix hj = acts.get("h"+j);
				deltaAlpha.put(j-bsIdx, deltaS.mmul(hj.transpose()).get(0));
			}
			// delta e
			DoubleMatrix alpha = acts.get("alpha"+t);
			DoubleMatrix deltaE = deltaAlpha.mul(alpha).sub(alpha.mmul(deltaAlpha.transpose()).mmul(alpha));
//            DoubleMatrix deltaE_ = new DoubleMatrix(t+1);
//            for(int j=0; j<t+1; j++) {
//            	deltaE_.put(j, deltaAlpha.get(j)*alpha.get(j)
//            					-deltaAlpha.transpose().mmul(alpha.mul(alpha.get(j))).get(0)
//            					);
//            }
			acts.put("de"+t, deltaE);
			// delta gs
			DoubleMatrix gs = acts.get("gs"+t);
//			DoubleMatrix deltaGs = new DoubleMatrix(t+1, outSize);
			DoubleMatrix deltaGs = new DoubleMatrix(eSize, outSize);
			for(int k=bsIdx; k<t+1; k++) {
				deltaGs.putRow(k-bsIdx, V.mul(deriveTanh(gs.getRow(k-bsIdx)))
						.mul(deltaE.get(k-bsIdx))
						);
			}
			acts.put("dgs"+t, deltaGs);
    	}
		
		calcWeightsGradient(acts, lastT);
	}
	
	private void calcWeightsGradient(Map<String, DoubleMatrix> acts, int lastT) {
		
		DoubleMatrix dV = new DoubleMatrix(V.rows, V.columns);
		DoubleMatrix dW = new DoubleMatrix(W.rows, W.columns);
		DoubleMatrix dU = new DoubleMatrix(U.rows, U.columns);
		DoubleMatrix dbs = new DoubleMatrix(bs.rows, bs.columns);
		
		for(int t=1; t<lastT+1; t++) {
			DoubleMatrix deltaE = acts.get("de"+t);
			DoubleMatrix deltaGs = acts.get("dgs"+t);
			
			int bsIdx = Math.max(0, t-AlgConsHSoftmax.windowSize+1);
			int eSize = Math.min(t+1, AlgConsHSoftmax.windowSize);
			
			DoubleMatrix prevS = null;
			if(t>0) {
				prevS = acts.get("s"+(t-1)).transpose();
			} else {
				prevS = DoubleMatrix.zeros(outSize);
			}
//			DoubleMatrix prevS = acts.get("s"+(t-1)).transpose();
			dW = dW.add(prevS.mmul(DoubleMatrix.ones(1,eSize)).mmul(deltaGs));

			DoubleMatrix gs = acts.get("gs"+t);
			for(int k=bsIdx; k<t+1; k++) {
				DoubleMatrix hk = acts.get("h"+k).transpose();
				dV = dV.add(gs.getRow(k-bsIdx).mul(deltaE.get(k-bsIdx)));
				dU = dU.add(hk.mmul(deltaGs.getRow(k-bsIdx)));
				dbs = dbs.add(deltaGs.getRow(k-bsIdx));
			}
		}
		
		acts.put("dV", dV);
		acts.put("dU", dU);
		acts.put("dW", dW);
		acts.put("dbs", dbs);
	}
	
	public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	AttBatchDerivative batchDerv = (AttBatchDerivative)derv;
    	
        hdV = hdV.add(MatrixFunctions.pow(batchDerv.dV, 2.));
        hdU = hdU.add(MatrixFunctions.pow(batchDerv.dU, 2.));
        hdW = hdW.add(MatrixFunctions.pow(batchDerv.dW, 2.));
        hdbs = hdbs.add(MatrixFunctions.pow(batchDerv.dbs, 2.));
        
        V = V.sub(batchDerv.dV.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdV).add(eps),-1.).mul(lr)));
        U = U.sub(batchDerv.dU.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdU).add(eps),-1.).mul(lr)));
        W = W.sub(batchDerv.dW.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdW).add(eps),-1.).mul(lr)));
        bs = bs.sub(batchDerv.dbs.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbs).add(eps),-1.).mul(lr)));
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
    						, double beta1, double beta2, int epochT) {
    	
    	AttBatchDerivative batchDerv = (AttBatchDerivative)derv;

		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		hd2V = hd2V.mul(beta2).add(MatrixFunctions.pow(batchDerv.dV, 2.).mul(1 - beta2));
		hd2U = hd2U.mul(beta2).add(MatrixFunctions.pow(batchDerv.dU, 2.).mul(1 - beta2));
		hd2W = hd2W.mul(beta2).add(MatrixFunctions.pow(batchDerv.dW, 2.).mul(1 - beta2));
		hd2bs = hd2bs.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbs, 2.).mul(1 - beta2));
		
		hdV = hdV.mul(beta1).add(batchDerv.dV.mul(1 - beta1));
		hdU = hdU.mul(beta1).add(batchDerv.dU.mul(1 - beta1));
		hdW = hdW.mul(beta1).add(batchDerv.dW.mul(1 - beta1));
		hdbs = hdbs.mul(beta1).add(batchDerv.dbs.mul(1 - beta1));

		V = V.sub(
					hdV.mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2V.mul(biasBeta2)).add(eps), -1))
					);
		U = U.sub(
				hdU.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2U.mul(biasBeta2)).add(eps), -1))
				);
		W = W.sub(
				hdW.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2W.mul(biasBeta2)).add(eps), -1))
				);
		bs = bs.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bs.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbs.mul(biasBeta1)).mul(lr)
				);
    }

	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#writeCellParameter(java.lang.String, boolean)
	 */
	@Override
	public void writeCellParameter(String outFile, boolean isAttached) {
		OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile, isAttached);
    	FileUtil.writeln(osw, "W");
    	writeMatrix(osw, W);
    	FileUtil.writeln(osw, "U");
    	writeMatrix(osw, U);
    	FileUtil.writeln(osw, "V");
    	writeMatrix(osw, V);
    	FileUtil.writeln(osw, "bs");
    	writeMatrix(osw, bs);
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
					String typeStr = "Null";
    				String[] typeList = {"W", "U", "V", "bs"};
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
					case W: this.W = matrixSetter(row, elems, this.W); break;
					case U: this.U = matrixSetter(row, elems, this.U); break;
					case V: this.V = matrixSetter(row, elems, this.V); break;
					case bs: this.bs = matrixSetter(row, elems, this.bs); break;
				}
				row++;
			}
			
		} catch(IOException e) {
			
		}
	}
}
