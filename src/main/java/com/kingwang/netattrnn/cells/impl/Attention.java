/**   
 * @package	com.kingwang.cdmrnn.rnn
 * @File		Attention.java
 * @Crtdate	Sep 28, 2016
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
import com.kingwang.netattrnn.batchderv.impl.AttBatchDerivative;
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
 * Sep 28, 2016 3:19:24 PM
 * @version 1.0
 */
public class Attention extends Operator implements Cell, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3581712325656989072L;
	
	public DoubleMatrix Wtt;
	public DoubleMatrix Wst;
	public DoubleMatrix Wxt;
	public DoubleMatrix Wdt;
	public DoubleMatrix bt;
	
	public DoubleMatrix hdWtt;
	public DoubleMatrix hdWst;
	public DoubleMatrix hdWxt;
	public DoubleMatrix hdWdt;
	public DoubleMatrix hdbt;
	
	public DoubleMatrix hd2Wtt;
	public DoubleMatrix hd2Wst;
	public DoubleMatrix hd2Wxt;
	public DoubleMatrix hd2Wdt;
	public DoubleMatrix hd2bt;
	
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
	
	private int hiddenSize = 0;
	private int attSize = 0;
	
	public Attention(int inDynSize, int inFixedSize, int attSize, int hiddenSize, MatIniter initer) {
		hdV = new DoubleMatrix(1, attSize);
		hdW = new DoubleMatrix(hiddenSize, attSize);
		hdU = new DoubleMatrix(attSize, attSize);
		hdbs = new DoubleMatrix(1, attSize);
		
		hd2V = new DoubleMatrix(1, attSize);
		hd2W = new DoubleMatrix(hiddenSize, attSize);
		hd2U = new DoubleMatrix(attSize, attSize);
		hd2bs = new DoubleMatrix(1, attSize);
		
		hdWtt = new DoubleMatrix(hiddenSize, hiddenSize);        
        hdWst = new DoubleMatrix(attSize, hiddenSize);
        hdWxt = new DoubleMatrix(inDynSize, hiddenSize);
        hdWdt = new DoubleMatrix(inFixedSize, hiddenSize);
        hdbt = new DoubleMatrix(1, hiddenSize);
        
        hd2Wtt = new DoubleMatrix(hiddenSize, hiddenSize);        
        hd2Wst = new DoubleMatrix(attSize, hiddenSize);
        hd2Wxt = new DoubleMatrix(inDynSize, hiddenSize);
        hd2Wdt = new DoubleMatrix(inFixedSize, hiddenSize);
        hd2bt = new DoubleMatrix(1, hiddenSize);
		
		if (initer.getType() == Type.Uniform) {
			Wtt = initer.uniform(hiddenSize, hiddenSize);
        	Wst = initer.uniform(attSize, hiddenSize);
        	Wxt = initer.uniform(inDynSize, hiddenSize);
        	Wdt = initer.uniform(inFixedSize, hiddenSize);
        	bt = initer.uniform(1, hiddenSize);
			
			V = initer.uniform(1, attSize);
			W = initer.uniform(hiddenSize, attSize);
			U = initer.uniform(attSize, attSize);
			bs = new DoubleMatrix(1, attSize).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
        	Wtt = initer.gaussian(hiddenSize, hiddenSize);
        	Wst = initer.gaussian(attSize, hiddenSize);
        	Wxt = initer.gaussian(inDynSize, hiddenSize);
        	Wdt = initer.gaussian(inFixedSize, hiddenSize);
        	bt = initer.gaussian(1, hiddenSize);
        	
        	V = initer.gaussian(1, attSize);
    		W = initer.gaussian(hiddenSize, attSize);
    		U = initer.gaussian(attSize, attSize);
    		bs = new DoubleMatrix(1, attSize).add(AlgConsHSoftmax.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
        	Wtt = initer.svd(hiddenSize, hiddenSize);
        	Wst = initer.svd(attSize, hiddenSize);
        	Wxt = initer.svd(inDynSize, hiddenSize);
        	Wdt = initer.svd(inFixedSize, hiddenSize);
        	bt = initer.svd(1, hiddenSize);
        	
        	V = initer.svd(1, attSize);
    		W = initer.svd(hiddenSize, attSize);
    		U = initer.svd(attSize, attSize);
    		bs = new DoubleMatrix(1, attSize).add(AlgConsHSoftmax.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        	V = new DoubleMatrix(1, attSize).add(0.1);
    		W = new DoubleMatrix(hiddenSize, attSize).add(0.1);
    		U = new DoubleMatrix(attSize, attSize).add(0.2);
    		bs = new DoubleMatrix(1, attSize).add(0.4);
        }
		
		this.hiddenSize = hiddenSize;
		this.attSize = attSize;
	}
	
	public void active(int t, Map<String, DoubleMatrix> acts, double... params) {
		
		DoubleMatrix prevT = null;
		if(t>0) {
			prevT = acts.get("t"+(t-1));
		} else {
			prevT = DoubleMatrix.zeros(1, hiddenSize);
		}
		int eSize = Math.min(t+1, AlgConsHSoftmax.windowSize);
		DoubleMatrix eWeight = new DoubleMatrix(eSize);
		
		DoubleMatrix gs = new DoubleMatrix(t+1, attSize);
		int bsIdx = Math.max(0, t-AlgConsHSoftmax.windowSize+1);
		for(int k=bsIdx; k<t+1; k++) {
			DoubleMatrix hk = acts.get("h"+k);
			DoubleMatrix gsk = Activer.tanh(prevT.mmul(W).add(hk.mmul(U)).add(bs));
			gs.putRow(k-bsIdx, gsk);
			eWeight.putRow(k-bsIdx, V.mmul(gsk.transpose()));
		}
		acts.put("gs"+t, gs);
		
		DoubleMatrix alpha = Activer.softmax(eWeight.transpose()).transpose();
		acts.put("alpha"+t, alpha);
		
		DoubleMatrix s = new DoubleMatrix(1, attSize);
		for(int k=bsIdx; k<t+1; k++) {
			DoubleMatrix hk = acts.get("h"+k);
			s = s.add(hk.mul(alpha.get(k-bsIdx)));
		}
		acts.put("s"+t, s);
		
    	DoubleMatrix x = acts.get("x"+t);
    	DoubleMatrix fixedFeat = acts.get("fixedFeat" + t);
    	
    	DoubleMatrix vecT = Activer.logistic(x.mmul(Wxt).add(fixedFeat.mmul(Wdt)).add(prevT.mmul(Wtt))
    							.add(s.mmul(Wst)).add(bt));
    	acts.put("t"+t, vecT);
		
	}
	
	public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {

		OutputLayerWithHSoftMax outLayer = (OutputLayerWithHSoftMax)cell[0];
		
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
			
    		// delta t
    		DoubleMatrix vecT = acts.get("t"+t);
            DoubleMatrix deltaT = deltaY.mmul(outLayer.Wty[cidx].transpose())
            						.add(deltaCls.mmul(outLayer.Wtc.transpose()));
            if(t<lastT) {
            	DoubleMatrix lateDat = acts.get("dAt"+(t+1));
            	DoubleMatrix lateDgs = acts.get("dgs"+(t+1));
            	int lateESize = Math.min(t+2, AlgConsHSoftmax.windowSize);
            	deltaT = deltaT.add(lateDat.mmul(Wtt.transpose()))
            					.add(DoubleMatrix.ones(1, lateESize).mmul(lateDgs.mmul(W.transpose())));
            }
            DoubleMatrix deltaAt = deltaT.mul(deriveExp(vecT));
            acts.put("dAt" + t, deltaAt);

            // delta s
            DoubleMatrix deltaS = deltaY.mmul(outLayer.Wsy[cidx].transpose())
            		.add(deltaCls.mmul(outLayer.Wsc.transpose()))
            		.add(deltaAt.mmul(Wst.transpose()));
            acts.put("ds"+t, deltaS);
			
			// delta alpha
            DoubleMatrix alpha = acts.get("alpha"+t);
            int bsIdx = Math.max(0, t-AlgConsHSoftmax.windowSize+1);
			int eSize = Math.min(t+1, AlgConsHSoftmax.windowSize);
			DoubleMatrix deltaAlpha = new DoubleMatrix(eSize);
			for(int j=bsIdx; j<t+1; j++) {
				DoubleMatrix hj = acts.get("h"+j);
				deltaAlpha.put(j-bsIdx, deltaS.mmul(hj.transpose()).get(0));
			}
			
			// delta e
			DoubleMatrix deltaE = deltaAlpha.mul(alpha).sub(alpha.mmul(deltaAlpha.transpose()).mmul(alpha));
			acts.put("de"+t, deltaE);
			
			// delta gs
			DoubleMatrix gs = acts.get("gs"+t);
			DoubleMatrix deltaGs = new DoubleMatrix(eSize, attSize);
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
		
		DoubleMatrix dWxt = new DoubleMatrix(Wxt.rows, Wxt.columns);
    	DoubleMatrix dWdt = new DoubleMatrix(Wdt.rows, Wdt.columns);
    	DoubleMatrix dWtt = new DoubleMatrix(Wtt.rows, Wtt.columns);
    	DoubleMatrix dWst = new DoubleMatrix(Wst.rows, Wst.columns);
    	DoubleMatrix dbt = new DoubleMatrix(bt.rows, bt.columns);
		
		DoubleMatrix dV = new DoubleMatrix(V.rows, V.columns);
		DoubleMatrix dW = new DoubleMatrix(W.rows, W.columns);
		DoubleMatrix dU = new DoubleMatrix(U.rows, U.columns);
		DoubleMatrix dbs = new DoubleMatrix(bs.rows, bs.columns);
		
		for(int t=0; t<lastT+1; t++) {
			DoubleMatrix deltaE = acts.get("de"+t);
			DoubleMatrix deltaGs = acts.get("dgs"+t);
			DoubleMatrix deltaAt = acts.get("dAt"+t);
			
			int bsIdx = Math.max(0, t-AlgConsHSoftmax.windowSize+1);
			int eSize = Math.min(t+1, AlgConsHSoftmax.windowSize);
			
			DoubleMatrix prevT = null;
			if(t>0) {
				prevT = acts.get("t"+(t-1)).transpose();
			} else {
				prevT = DoubleMatrix.zeros(hiddenSize);
			}
//			DoubleMatrix prevS = acts.get("s"+(t-1)).transpose();
			dW = dW.add(prevT.mmul(DoubleMatrix.ones(1,eSize)).mmul(deltaGs));

			DoubleMatrix gs = acts.get("gs"+t);
			for(int k=bsIdx; k<t+1; k++) {
				DoubleMatrix hk = acts.get("h"+k).transpose();
				dV = dV.add(gs.getRow(k-bsIdx).mul(deltaE.get(k-bsIdx)));
				dU = dU.add(hk.mmul(deltaGs.getRow(k-bsIdx)));
				dbs = dbs.add(deltaGs.getRow(k-bsIdx));
			}
			
			DoubleMatrix x = acts.get("x" + t).transpose();
            DoubleMatrix fixedFeat = acts.get("fixedFeat" + t).transpose();
            DoubleMatrix s = acts.get("s" + t).transpose();
            
			dWxt = dWxt.add(x.mmul(deltaAt));
            dWdt = dWdt.add(fixedFeat.mmul(deltaAt));
            dWtt = dWtt.add(prevT.mmul(deltaAt));
            dWst = dWst.add(s.mmul(deltaAt));
            dbt = dbt.add(deltaAt);
		}
		
		acts.put("dWxt", dWxt);
    	acts.put("dWdt", dWdt);
    	acts.put("dWtt", dWtt);
    	acts.put("dWst", dWst);
    	acts.put("dbt", dbt);
		
		acts.put("dV", dV);
		acts.put("dU", dU);
		acts.put("dW", dW);
		acts.put("dbs", dbs);
	}
	
	public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	AttBatchDerivative batchDerv = (AttBatchDerivative)derv;
    	
    	hdWxt = hdWxt.add(MatrixFunctions.pow(batchDerv.dWxt, 2.));
    	hdWdt = hdWdt.add(MatrixFunctions.pow(batchDerv.dWdt, 2.));
    	hdWtt = hdWtt.add(MatrixFunctions.pow(batchDerv.dWtt, 2.));
    	hdWst = hdWst.add(MatrixFunctions.pow(batchDerv.dWst, 2.));
    	hdbt = hdbt.add(MatrixFunctions.pow(batchDerv.dbt, 2.));
    	
        hdV = hdV.add(MatrixFunctions.pow(batchDerv.dV, 2.));
        hdU = hdU.add(MatrixFunctions.pow(batchDerv.dU, 2.));
        hdW = hdW.add(MatrixFunctions.pow(batchDerv.dW, 2.));
        hdbs = hdbs.add(MatrixFunctions.pow(batchDerv.dbs, 2.));
        
        Wxt = Wxt.sub(batchDerv.dWxt.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxt).add(eps),-1.).mul(lr)));
		Wdt = Wdt.sub(batchDerv.dWdt.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdt).add(eps),-1.).mul(lr)));
		Wtt = Wtt.sub(batchDerv.dWtt.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWtt).add(eps),-1.).mul(lr)));
		Wst = Wst.sub(batchDerv.dWst.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdWst).add(eps),-1.).mul(lr)));
		bt = bt.sub(batchDerv.dbt.mul(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hdbt).add(eps),-1.).mul(lr)));
        
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

		hd2Wxt = hd2Wxt.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxt, 2.).mul(1 - beta2));
		hd2Wdt = hd2Wdt.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdt, 2.).mul(1 - beta2));
		hd2Wtt = hd2Wtt.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWtt, 2.).mul(1 - beta2));
		hd2Wst = hd2Wst.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWst, 2.).mul(1 - beta2));
		hd2bt = hd2bt.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbt, 2.).mul(1 - beta2));
		
		hdWxt = hdWxt.mul(beta1).add(batchDerv.dWxt.mul(1 - beta1));
		hdWdt = hdWdt.mul(beta1).add(batchDerv.dWdt.mul(1 - beta1));
		hdWtt = hdWtt.mul(beta1).add(batchDerv.dWtt.mul(1 - beta1));
		hdWst = hdWst.mul(beta1).add(batchDerv.dWst.mul(1 - beta1));
		hdbt = hdbt.mul(beta1).add(batchDerv.dbt.mul(1 - beta1));
		
		hd2V = hd2V.mul(beta2).add(MatrixFunctions.pow(batchDerv.dV, 2.).mul(1 - beta2));
		hd2U = hd2U.mul(beta2).add(MatrixFunctions.pow(batchDerv.dU, 2.).mul(1 - beta2));
		hd2W = hd2W.mul(beta2).add(MatrixFunctions.pow(batchDerv.dW, 2.).mul(1 - beta2));
		hd2bs = hd2bs.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbs, 2.).mul(1 - beta2));
		
		hdV = hdV.mul(beta1).add(batchDerv.dV.mul(1 - beta1));
		hdU = hdU.mul(beta1).add(batchDerv.dU.mul(1 - beta1));
		hdW = hdW.mul(beta1).add(batchDerv.dW.mul(1 - beta1));
		hdbs = hdbs.mul(beta1).add(batchDerv.dbs.mul(1 - beta1));

		Wxt = Wxt.sub(
				hdWxt.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxt.mul(biasBeta2)).add(eps), -1))
				);
		Wdt = Wdt.sub(
				hdWdt.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdt.mul(biasBeta2)).add(eps), -1))
				);
		Wtt = Wtt.sub(
				hdWtt.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wtt.mul(biasBeta2)).add(eps), -1))
				);
		Wst = Wst.sub(
				hdWst.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wst.mul(biasBeta2)).add(eps), -1))
				);
		bt = bt.sub(
				hdbt.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bt.mul(biasBeta2)).add(eps), -1))
				);
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
		FileUtil.writeln(osw, "Wxt");
		writeMatrix(osw, Wxt);
		FileUtil.writeln(osw, "Wdt");
		writeMatrix(osw, Wdt);
		FileUtil.writeln(osw, "Wtt");
		writeMatrix(osw, Wtt);
		FileUtil.writeln(osw, "Wst");
		writeMatrix(osw, Wst);
		FileUtil.writeln(osw, "bt");
		writeMatrix(osw, bt);
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
    				String[] typeList = {"W", "U", "V", "bs", "Wxt", "Wdt", "Wtt", "Wst", "bt"};
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
					case Wxt: this.Wxt = matrixSetter(row, elems, this.Wxt); break;
					case Wdt: this.Wdt = matrixSetter(row, elems, this.Wdt); break;
					case Wtt: this.Wtt = matrixSetter(row, elems, this.Wtt); break;
					case Wst: this.Wst = matrixSetter(row, elems, this.Wst); break;
					case bt: this.bt = matrixSetter(row, elems, this.bt); break;
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
