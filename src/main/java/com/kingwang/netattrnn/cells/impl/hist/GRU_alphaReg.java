package com.kingwang.netattrnn.cells.impl.hist;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.netattrnn.batchderv.BatchDerivative;
import com.kingwang.netattrnn.batchderv.impl.GRUBatchDerivative;
import com.kingwang.netattrnn.cells.Cell;
import com.kingwang.netattrnn.cells.Operator;
import com.kingwang.netattrnn.comm.utils.FileUtil;
import com.kingwang.netattrnn.cons.AlgConsHSoftmax;
import com.kingwang.netattrnn.utils.Activer;
import com.kingwang.netattrnn.utils.LoadTypes;
import com.kingwang.netattrnn.utils.MatIniter;
import com.kingwang.netattrnn.utils.MatIniter.Type;

public class GRU_alphaReg extends Operator implements Cell, Serializable{
	
    private static final long serialVersionUID = -1501734916541393551L;

    private int outSize;
    
    /**
     * historical first-derivative gradient of weights 
     */
    public DoubleMatrix hdWxr;
    public DoubleMatrix hdWdr;
    public DoubleMatrix hdWhr;
    public DoubleMatrix hdbr;
    
    public DoubleMatrix hdWxz;
    public DoubleMatrix hdWdz;
    public DoubleMatrix hdWhz;
    public DoubleMatrix hdbz;
    
    public DoubleMatrix hdWxh;
    public DoubleMatrix hdWdh;
    public DoubleMatrix hdWhh;
    public DoubleMatrix hdbh;
    
    /**
     * historical second-derivative gradient of weights 
     */
    public DoubleMatrix hd2Wxr;
    public DoubleMatrix hd2Wdr;
    public DoubleMatrix hd2Whr;
    public DoubleMatrix hd2br;
    
    public DoubleMatrix hd2Wxz;
    public DoubleMatrix hd2Wdz;
    public DoubleMatrix hd2Whz;
    public DoubleMatrix hd2bz;
    
    public DoubleMatrix hd2Wxh;
    public DoubleMatrix hd2Wdh;
    public DoubleMatrix hd2Whh;
    public DoubleMatrix hd2bh;
    
    /**
     * model parameters
     */
    public DoubleMatrix Wxr;
    public DoubleMatrix Wdr;
    public DoubleMatrix Whr;
    public DoubleMatrix br;
    
    public DoubleMatrix Wxz;
    public DoubleMatrix Wdz;
    public DoubleMatrix Whz;
    public DoubleMatrix bz;
    
    public DoubleMatrix Wxh;
    public DoubleMatrix Wdh;
    public DoubleMatrix Whh;
    public DoubleMatrix bh;
    
    public GRU_alphaReg(int inDynSize, int inFixedSize, int outSize, MatIniter initer) {
        this.outSize = outSize;
        
        if (initer.getType() == Type.Uniform) {
            this.Wxr = initer.uniform(inDynSize, outSize);
            this.Wdr = initer.uniform(inFixedSize, outSize);
            this.Whr = initer.uniform(outSize, outSize);
            
            this.Wxz = initer.uniform(inDynSize, outSize);
            this.Wdz = initer.uniform(inFixedSize, outSize);
            this.Whz = initer.uniform(outSize, outSize);
            
            this.Wxh = initer.uniform(inDynSize, outSize);
            this.Wdh = initer.uniform(inFixedSize, outSize);
            this.Whh = initer.uniform(outSize, outSize);
            
        } else if (initer.getType() == Type.Gaussian) {
        	this.Wxr = initer.gaussian(inDynSize, outSize);
            this.Wdr = initer.gaussian(inFixedSize, outSize);
            this.Whr = initer.gaussian(outSize, outSize);
            
            this.Wxz = initer.gaussian(inDynSize, outSize);
            this.Wdz = initer.gaussian(inFixedSize, outSize);
            this.Whz = initer.gaussian(outSize, outSize);
            
            this.Wxh = initer.gaussian(inDynSize, outSize);
            this.Wdh = initer.gaussian(inFixedSize, outSize);
            this.Whh = initer.gaussian(outSize, outSize);
        } else if (initer.getType() == Type.SVD) {
        	this.Wxr = initer.svd(inDynSize, outSize);
            this.Wdr = initer.svd(inFixedSize, outSize);
            this.Whr = initer.svd(outSize, outSize);
            
            this.Wxz = initer.svd(inDynSize, outSize);
            this.Wdz = initer.svd(inFixedSize, outSize);
            this.Whz = initer.svd(outSize, outSize);
            
            this.Wxh = initer.svd(inDynSize, outSize);
            this.Wdh = initer.svd(inFixedSize, outSize);
            this.Whh = initer.svd(outSize, outSize);
        } else if(initer.getType() == Type.Test) {
        	this.Wxr = DoubleMatrix.ones(inDynSize, outSize).mul(0.1);
        	this.Wdr = DoubleMatrix.zeros(inFixedSize, outSize).add(0.1);
            this.Whr = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.br = DoubleMatrix.zeros(1, outSize).add(0.4);
            
            this.Wxz = DoubleMatrix.zeros(inDynSize, outSize).add(0.1);
            this.Wdz = DoubleMatrix.zeros(inFixedSize, outSize).add(0.1);
            this.Whz = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.bz = DoubleMatrix.zeros(1, outSize).add(0.4);
            
            this.Wxh = DoubleMatrix.zeros(inDynSize, outSize).add(0.1);
            this.Wdh = DoubleMatrix.zeros(inFixedSize, outSize).add(0.1);
            this.Whh = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.bh = DoubleMatrix.zeros(1, outSize).add(0.3);
        }
        this.br = new DoubleMatrix(1, outSize).add(AlgConsHSoftmax.biasInitVal);
        this.bz = new DoubleMatrix(1, outSize).add(AlgConsHSoftmax.biasInitVal);
        this.bh = new DoubleMatrix(1, outSize).add(AlgConsHSoftmax.biasInitVal);
        
        this.hdWxr = new DoubleMatrix(inDynSize, outSize);
        this.hdWdr = new DoubleMatrix(inFixedSize, outSize);
        this.hdWhr = new DoubleMatrix(outSize, outSize);
        this.hdbr = new DoubleMatrix(1, outSize);
        
        this.hdWxz = new DoubleMatrix(inDynSize, outSize);
        this.hdWdz = new DoubleMatrix(inFixedSize, outSize);
        this.hdWhz = new DoubleMatrix(outSize, outSize);
        this.hdbz = new DoubleMatrix(1, outSize);
        
        this.hdWxh = new DoubleMatrix(inDynSize, outSize);
        this.hdWdh = new DoubleMatrix(inFixedSize, outSize);
        this.hdWhh = new DoubleMatrix(outSize, outSize);
        this.hdbh = new DoubleMatrix(1, outSize);
        
        this.hd2Wxr = new DoubleMatrix(inDynSize, outSize);
        this.hd2Wdr = new DoubleMatrix(inFixedSize, outSize);
        this.hd2Whr = new DoubleMatrix(outSize, outSize);
        this.hd2br = new DoubleMatrix(1, outSize);
        
        this.hd2Wxz = new DoubleMatrix(inDynSize, outSize);
        this.hd2Wdz = new DoubleMatrix(inFixedSize, outSize);
        this.hd2Whz = new DoubleMatrix(outSize, outSize);
        this.hd2bz = new DoubleMatrix(1, outSize);
        
        this.hd2Wxh = new DoubleMatrix(inDynSize, outSize);
        this.hd2Wdh = new DoubleMatrix(inFixedSize, outSize);
        this.hd2Whh = new DoubleMatrix(outSize, outSize);
        this.hd2bh = new DoubleMatrix(1, outSize);
    }
    
    public void active(int t, Map<String, DoubleMatrix> acts, double... params) {
        DoubleMatrix x = acts.get("x" + t);
        DoubleMatrix fixedFeat = acts.get("fixedFeat" + t);
        
        DoubleMatrix preH = null;
        if (t == 0) {
            preH = new DoubleMatrix(1, outSize);
        } else {
            preH = acts.get("h" + (t - 1));
        }
        
        DoubleMatrix r = Activer.logistic(x.mmul(Wxr).add(fixedFeat.mmul(Wdr)).add(preH.mmul(Whr)).add(br));
        DoubleMatrix z = Activer.logistic(x.mmul(Wxz).add(fixedFeat.mmul(Wdz)).add(preH.mmul(Whz)).add(bz));
        DoubleMatrix gh = Activer.tanh(x.mmul(Wxh).add(fixedFeat.mmul(Wdh)).add(r.mul(preH.mmul(Whh))).add(bh));
        DoubleMatrix h = z.mul(preH).add((DoubleMatrix.ones(1, z.columns).sub(z)).mul(gh));
        
        acts.put("r" + t, r);
        acts.put("z" + t, z);
        acts.put("gh" + t, gh);
        acts.put("h" + t, h);
    }
    
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {
    	
    	Attention_alphaReg att = (Attention_alphaReg)cell[0];
    	
        for (int t = lastT; t > -1; t--) {
            // cell output errors
            DoubleMatrix h = acts.get("h" + t);
            DoubleMatrix z = acts.get("z" + t);
            DoubleMatrix r = acts.get("r" + t);
            DoubleMatrix gh = acts.get("gh" + t);
            
            DoubleMatrix deltaH = new DoubleMatrix(h.rows, h.columns);
            int bsIdx = Math.max(0, t-AlgConsHSoftmax.windowSize+1);
            int maxSize = Math.min(lastT+1, t+AlgConsHSoftmax.windowSize);
            for(int k=t; k<maxSize; k++) {
            	DoubleMatrix deltaS = acts.get("ds"+k);
            	DoubleMatrix deltaGs = acts.get("dgs"+k);
            	DoubleMatrix alpha = acts.get("alpha"+k);
            	deltaH = deltaH.add(deltaS.mul(alpha.get(t-bsIdx))
            						.add(deltaGs.getRow(t-bsIdx).mmul(att.U.transpose())));
            }
            
            if (t < lastT) {
                DoubleMatrix lateDh = acts.get("dh"+(t+1));
                DoubleMatrix lateDgh = acts.get("dgh"+(t+1));
                DoubleMatrix lateDr = acts.get("dr"+(t+1));
                DoubleMatrix lateDz = acts.get("dz"+(t+1));
                DoubleMatrix lateR = acts.get("r"+(t+1));
                DoubleMatrix lateZ = acts.get("z"+(t+1));
                deltaH = deltaH.add(lateDr.mmul(Whr.transpose()))
                        .add(lateDz.mmul(Whz.transpose()))
                        .add(lateDgh.mul(lateR).mmul(Whh.transpose()))
                        .add(lateDh.mul(lateZ));
            }
            acts.put("dh" + t, deltaH);
            
            // gh
            DoubleMatrix deltaGh = deltaH.mul(DoubleMatrix.ones(1, z.columns).sub(z)).mul(deriveTanh(gh));
            acts.put("dgh" + t, deltaGh);
            
            DoubleMatrix preH = null;
            if (t > 0) {
                preH = acts.get("h" + (t - 1));
            } else {
                preH = DoubleMatrix.zeros(1, h.length);
            }
            
            // reset gates
            DoubleMatrix deltaR = preH.mmul(Whh).mul(deltaGh).mul(deriveExp(r));
            acts.put("dr" + t, deltaR);
            
            // update gates
            DoubleMatrix deltaZ = deltaH.mul(preH.sub(gh)).mul(deriveExp(z));
            acts.put("dz" + t, deltaZ);
        }
        calcWeightsGradient(acts, lastT);
    }
    
    private void calcWeightsGradient(Map<String, DoubleMatrix> acts, int lastT) {
        DoubleMatrix dWxr = new DoubleMatrix(Wxr.rows, Wxr.columns);
        DoubleMatrix dWdr = new DoubleMatrix(Wdr.rows, Wdr.columns);
        DoubleMatrix dWhr = new DoubleMatrix(Whr.rows, Whr.columns);
        DoubleMatrix dbr = new DoubleMatrix(br.rows, br.columns);
        
        DoubleMatrix dWxz = new DoubleMatrix(Wxz.rows, Wxz.columns);
        DoubleMatrix dWdz = new DoubleMatrix(Wdz.rows, Wdz.columns);
        DoubleMatrix dWhz = new DoubleMatrix(Whz.rows, Whz.columns);
        DoubleMatrix dbz = new DoubleMatrix(bz.rows, bz.columns);
        
        DoubleMatrix dWxh = new DoubleMatrix(Wxh.rows, Wxh.columns);
        DoubleMatrix dWdh = new DoubleMatrix(Wdh.rows, Wdh.columns);
        DoubleMatrix dWhh = new DoubleMatrix(Whh.rows, Whh.columns);
        DoubleMatrix dbh = new DoubleMatrix(bh.rows, bh.columns);
        
        for (int t = 0; t < lastT + 1; t++) {
        	DoubleMatrix x = acts.get("x" + t).transpose();
            DoubleMatrix tmFeat = acts.get("fixedFeat" + t).transpose();
            
            dWxr = dWxr.add(x.mmul(acts.get("dr" + t)));
            dWxz = dWxz.add(x.mmul(acts.get("dz" + t)));
            dWxh = dWxh.add(x.mmul(acts.get("dgh" + t)));
            
            dWdr = dWdr.add(tmFeat.mmul(acts.get("dr" + t)));
            dWdz = dWdz.add(tmFeat.mmul(acts.get("dz" + t)));
            dWdh = dWdh.add(tmFeat.mmul(acts.get("dgh" + t)));
            
            if (t > 0) {
                DoubleMatrix preH = acts.get("h" + (t - 1)).transpose();
                dWhr = dWhr.add(preH.mmul(acts.get("dr" + t)));
                dWhz = dWhz.add(preH.mmul(acts.get("dz" + t)));
                dWhh = dWhh.add(preH.mmul(acts.get("r" + t).mul(acts.get("dgh" + t))));
            }
            
            dbr = dbr.add(acts.get("dr" + t));
            dbz = dbz.add(acts.get("dz" + t));
            dbh = dbh.add(acts.get("dgh" + t));
        }
        
        acts.put("dWxr", dWxr);
        acts.put("dWdr", dWdr);
        acts.put("dWhr", dWhr);
        acts.put("dbr", dbr);
        
        acts.put("dWxz", dWxz);
        acts.put("dWdz", dWdz);
        acts.put("dWhz", dWhz);
        acts.put("dbz", dbz);
        
        acts.put("dWxh", dWxh);
        acts.put("dWdh", dWdh);
        acts.put("dWhh", dWhh);
        acts.put("dbh", dbh);
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	GRUBatchDerivative batchDerv = (GRUBatchDerivative) derv;
    	
        hdWxr = hdWxr.add(MatrixFunctions.pow(batchDerv.dWxr, 2.));
        hdWdr = hdWdr.add(MatrixFunctions.pow(batchDerv.dWdr, 2.));
        hdWhr = hdWhr.add(MatrixFunctions.pow(batchDerv.dWhr, 2.));
        hdbr = hdbr.add(MatrixFunctions.pow(batchDerv.dbr, 2.));
        
        hdWxz = hdWxz.add(MatrixFunctions.pow(batchDerv.dWxz, 2.));
        hdWdz = hdWdz.add(MatrixFunctions.pow(batchDerv.dWdz, 2.));
        hdWhz = hdWhz.add(MatrixFunctions.pow(batchDerv.dWhz, 2.));
        hdbz = hdbz.add(MatrixFunctions.pow(batchDerv.dbz, 2.));
        
        hdWxh = hdWxh.add(MatrixFunctions.pow(batchDerv.dWxh, 2.));
        hdWdh = hdWdh.add(MatrixFunctions.pow(batchDerv.dWdh, 2.));
        hdWhh = hdWhh.add(MatrixFunctions.pow(batchDerv.dWhh, 2.));
        hdbh = hdbh.add(MatrixFunctions.pow(batchDerv.dbh, 2.));
        
        Wxr = Wxr.sub(batchDerv.dWxr.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxr).add(eps),-1.).mul(lr)));
        Wdr = Wdr.sub(batchDerv.dWdr.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdr).add(eps),-1.).mul(lr)));
        Whr = Whr.sub(batchDerv.dWhr.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhr).add(eps),-1.).mul(lr)));
        br = br.sub(batchDerv.dbr.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbr).add(eps),-1.).mul(lr)));
        
        Wxz = Wxz.sub(batchDerv.dWxz.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxz).add(eps),-1.).mul(lr)));
        Wdz = Wdz.sub(batchDerv.dWdz.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdz).add(eps),-1.).mul(lr)));
        Whz = Whz.sub(batchDerv.dWhz.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhz).add(eps),-1.).mul(lr)));
        bz = bz.sub(batchDerv.dbz.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbz).add(eps),-1.).mul(lr)));
        
        Wxh = Wxh.sub(batchDerv.dWxh.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxh).add(eps),-1.).mul(lr)));
        Wdh = Wdh.sub(batchDerv.dWdh.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdh).add(eps),-1.).mul(lr)));
        Whh = Whh.sub(batchDerv.dWhh.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhh).add(eps),-1.).mul(lr)));
        bh = bh.sub(batchDerv.dbh.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbh).add(eps),-1.).mul(lr)));
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
    						, double beta1, double beta2, int epochT) {
    	
    	GRUBatchDerivative batchDerv = (GRUBatchDerivative) derv;

		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		hd2Wxr = hd2Wxr.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxr, 2.).mul(1 - beta2));
		hd2Wdr = hd2Wdr.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdr, 2.).mul(1 - beta2));
		hd2Whr = hd2Whr.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhr, 2.).mul(1 - beta2));
		hd2br = hd2br.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbr, 2.).mul(1 - beta2));
		
		hdWxr = hdWxr.mul(beta1).add(batchDerv.dWxr.mul(1 - beta1));
		hdWdr = hdWdr.mul(beta1).add(batchDerv.dWdr.mul(1 - beta1));
		hdWhr = hdWhr.mul(beta1).add(batchDerv.dWhr.mul(1 - beta1));
		hdbr = hdbr.mul(beta1).add(batchDerv.dbr.mul(1 - beta1));

		hd2Wxz = hd2Wxz.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxz, 2.).mul(1 - beta2));
		hd2Wdz = hd2Wdz.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdz, 2.).mul(1 - beta2));
		hd2Whz = hd2Whz.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhz, 2.).mul(1 - beta2));
		hd2bz = hd2bz.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbz, 2.).mul(1 - beta2));
		
		hdWxz = hdWxz.mul(beta1).add(batchDerv.dWxz.mul(1 - beta1));
		hdWdz = hdWdz.mul(beta1).add(batchDerv.dWdz.mul(1 - beta1));
		hdWhz = hdWhz.mul(beta1).add(batchDerv.dWhz.mul(1 - beta1));
		hdbz = hdbz.mul(beta1).add(batchDerv.dbz.mul(1 - beta1));

		hd2Wxh = hd2Wxh.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxh, 2.).mul(1 - beta2));
		hd2Wdh = hd2Wdh.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdh, 2.).mul(1 - beta2));
		hd2Whh = hd2Whh.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhh, 2.).mul(1 - beta2));
		hd2bh = hd2bh.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbh, 2.).mul(1 - beta2));
		
		hdWxh = hdWxh.mul(beta1).add(batchDerv.dWxh.mul(1 - beta1));
		hdWdh = hdWdh.mul(beta1).add(batchDerv.dWdh.mul(1 - beta1));
		hdWhh = hdWhh.mul(beta1).add(batchDerv.dWhh.mul(1 - beta1));
		hdbh = hdbh.mul(beta1).add(batchDerv.dbh.mul(1 - beta1));

		Wxr = Wxr.sub(
					hdWxr.mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxr.mul(biasBeta2)).add(eps), -1))
					);
		Wdr = Wdr.sub(
				hdWdr.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdr.mul(biasBeta2)).add(eps), -1))
				);
		Whr = Whr.sub(
				hdWhr.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whr.mul(biasBeta2)).add(eps), -1))
				);
		br = br.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2br.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbr.mul(biasBeta1)).mul(lr)
				);
		
		Wxz = Wxz.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxz.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWxz.mul(biasBeta1)).mul(lr)
				);
		Wdz = Wdz.sub(
				hdWdz.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdz.mul(biasBeta2)).add(eps), -1))
				);
		Whz = Whz.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whz.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWhz.mul(biasBeta1)).mul(lr)
				);
		bz = bz.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bz.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbz.mul(biasBeta1)).mul(lr)
				);

		Wxh = Wxh.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxh.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWxh.mul(biasBeta1)).mul(lr)
				);
		Wdh = Wdh.sub(
				hdWdh.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdh.mul(biasBeta2)).add(eps), -1))
				);
		Whh = Whh.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whh.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWhh.mul(biasBeta1)).mul(lr)
				);
		bh = bh.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bh.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbh.mul(biasBeta1)).mul(lr)
				);
    }
    
	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#writeCellParameter(java.lang.String, boolean)
	 */
	@Override
	public void writeCellParameter(String outFile, boolean isAttached) {
		OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile, isAttached);
    	FileUtil.writeln(osw, "Wxr");
    	writeMatrix(osw, Wxr);
    	FileUtil.writeln(osw, "Wdr");
    	writeMatrix(osw, Wdr);
    	FileUtil.writeln(osw, "Whr");
    	writeMatrix(osw, Whr);
    	FileUtil.writeln(osw, "br");
    	writeMatrix(osw, br);
    	
    	FileUtil.writeln(osw, "Wxz");
    	writeMatrix(osw, Wxz);
    	FileUtil.writeln(osw, "Wdz");
    	writeMatrix(osw, Wdz);
    	FileUtil.writeln(osw, "Whz");
    	writeMatrix(osw, Whz);
    	FileUtil.writeln(osw, "bz");
    	writeMatrix(osw, bz);
    	
    	FileUtil.writeln(osw, "Wxh");
    	writeMatrix(osw, Wxh);
    	FileUtil.writeln(osw, "Wdh");
    	writeMatrix(osw, Wdh);
    	FileUtil.writeln(osw, "Whh");
    	writeMatrix(osw, Whh);
    	FileUtil.writeln(osw, "bh");
    	writeMatrix(osw, bh);
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
    				String[] typeList = {"Wxr", "Wdr", "Whr", "br", "Wxz", "Wdz", "Whz", "bz"
    									, "Wxh", "Wdh", "Whh", "bh"};
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
	    			case Wxr: this.Wxr = matrixSetter(row, elems, this.Wxr); break;
	    			case Wdr: this.Wdr = matrixSetter(row, elems, this.Wdr); break;
	    			case Whr: this.Whr = matrixSetter(row, elems, this.Whr); break;
	    			case br: this.br = matrixSetter(row, elems, this.br); break;
	    			
	    			case Wxz: this.Wxz = matrixSetter(row, elems, this.Wxz); break;
	    			case Wdz: this.Wdz = matrixSetter(row, elems, this.Wdz); break;
	    			case Whz: this.Whz = matrixSetter(row, elems, this.Whz); break;
	    			case bz: this.bz = matrixSetter(row, elems, this.bz); break;
	    			
	    			case Wxh: this.Wxh = matrixSetter(row, elems, this.Wxh); break;
	    			case Wdh: this.Wdh = matrixSetter(row, elems, this.Wdh); break;
	    			case Whh: this.Whh = matrixSetter(row, elems, this.Whh); break;
	    			case bh: this.bh = matrixSetter(row, elems, this.bh); break;
    			}
    			row++;
    		}
    		
    	} catch(IOException e) {
    		
    	}
	}
}
