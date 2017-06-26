/**   
 * @package	com.kingwang.rnncdm.evals
 * @File		RNNModelMRREvals.java
 * @Crtdate	May 22, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn.evals;

import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.jblas.DoubleMatrix;

import com.kingwang.netrnn.cells.impl.Attention;
import com.kingwang.netrnn.cells.impl.GRU;
import com.kingwang.netrnn.cells.impl.InputLayer;
import com.kingwang.netrnn.cells.impl.OutputLayerWithHSoftMax;
import com.kingwang.netrnn.comm.utils.CollectionHelper;
import com.kingwang.netrnn.comm.utils.FileUtil;
import com.kingwang.netrnn.comm.utils.StringHelper;
import com.kingwang.netrnn.cons.AlgConsHSoftmax;
import com.kingwang.netrnn.cons.MultiThreadCons;
import com.kingwang.netrnn.dataset.DataLoader;
import com.kingwang.netrnn.dataset.Node4Code;
import com.kingwang.netrnn.dataset.SeqLoader;
import com.kingwang.netrnn.utils.Activer;
import com.kingwang.netrnn.utils.InputEncoder;
import com.kingwang.netrnn.utils.LossFunction;
import com.kingwang.netrnn.utils.TmFeatExtractor;

/**
 *
 * @author King Wang
 * 
 * May 22, 2016 5:03:33 PM
 * @version 1.0
 */
public class CyanRNNModelEvals {
	
	public static Double logLkHd = .0;
	public static Double mrr = .0;
	public InputLayer input;
	public GRU gru;
	public Attention att;
	public OutputLayerWithHSoftMax output;
	public DataLoader casLoader;
	public OutputStreamWriter oswLog;
	
	public CyanRNNModelEvals(InputLayer input, GRU gru, Attention att, OutputLayerWithHSoftMax output
						, DataLoader casLoader, OutputStreamWriter oswLog) {
		this.input = input;
		this.gru = gru;
		this.att = att;
		this.output = output;
		this.casLoader = casLoader;
		this.oswLog = oswLog;
	}
	
	private void calcGradientByMiniBatch(List<String> sequence) {
		
    	MultiThreadCons.missions = getMissions(sequence);
    	MultiThreadCons.missionSize = sequence.size();
    	MultiThreadCons.missionOver = 0;
    	
		ExecutorService exec = Executors.newCachedThreadPool();
		for (int i = 0; i < MultiThreadCons.threadNum; i++) {
			exec.execute(new Exec());
		}
		while (MultiThreadCons.missionOver!=MultiThreadCons.threadNum) {
			try {
				Thread.sleep((long) (1000 * MultiThreadCons.sleepSec));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		exec.shutdown();
		try {
			exec.awaitTermination(500, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private List<String> getMissions(List<String> sequence) {
    	
    	List<String> missions = new ArrayList<>();
    	for(String seq : sequence) {
    		missions.add(seq);
    	}
    	
    	return missions;
    }
	
	public double validationOnIntegration() {

		List<String> crsValSeq = casLoader.getCrsValSeq();
		logLkHd = .0;
		mrr = .0;
    	calcGradientByMiniBatch(crsValSeq);
		
		logLkHd /= crsValSeq.size();
		mrr /= crsValSeq.size();
		System.out.println("The likelihood in Validation: " + logLkHd);
		FileUtil.writeln(oswLog, "The likelihood in Validation: " + logLkHd);
		System.out.println("The MRR of node prediction in Validation: " + mrr);
		FileUtil.writeln(oswLog, "The MRR of node prediction in Validation: " + mrr);

		return logLkHd;
	}
	
	class Exec implements Runnable {

		private void mainProc(String seq) {
			
			Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
			List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq);
			if(infos.size()<3) { //skip short cascades
            	return;
            }
            String iid = infos.remove(0);
            double cas_logLkHd=0, cas_mrr=0, prevTm=0;
			int missCnt = 0;
			for (int t = 0; t < infos.size() - 1; t++) {
				String[] curInfo = infos.get(t).split(",");
				String[] nextInfo = infos.get(t + 1).split(",");
				// translating string node to node index in repMatrix
				String curNd = curInfo[0];
            	String nxtNd = nextInfo[0];
            	double curTm = Double.parseDouble(curInfo[1]);
            	if(!casLoader.getCodeMaps().containsKey(curNd)) {//if curNd isn't located in nodeDict
//	            		System.out.println("Missing node"+curNd);
            		missCnt++;
            		curNd = "null";
            		break;//TODO: how to solve "null" node
            	}
            	if(!casLoader.getCodeMaps().containsKey(nxtNd)) {//if curNd isn't located in nodeDict
            		curNd = "null";
            		break;//TODO: how to solve "null" node
            	}
            	Node4Code nxtNd4Code = casLoader.getCodeMaps().get(nxtNd);
            	//Set DoubleMatrix code & fixedFeat. It should be a code setter function here.
            	DoubleMatrix tmFeat = TmFeatExtractor.timeFeatExtractor(curTm, prevTm);
            	DoubleMatrix fixedFeat;
				try {
					fixedFeat = InputEncoder.setFixedFeat(t, AlgConsHSoftmax.inFixedSize, tmFeat);
					acts.put("fixedFeat"+t, fixedFeat);
//					DoubleMatrix code = new DoubleMatrix(1, AlgConsHSoftmax.nodeSize); 
//					code = InputEncoder.setBinaryCode(curNd, code);
					DoubleMatrix code = new DoubleMatrix(1);
					code.put(0, Double.parseDouble(curNd));
					acts.put("code"+t, code);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					break;
				}
				
				int nodeCls = nxtNd4Code.nodeCls;
            	
            	input.active(t, acts);
                gru.active(t, acts);
                att.active(t, acts);
                output.active(t, acts, nodeCls);
            	
                //actual u
                int nxtNdIdxInCls = nxtNd4Code.idxInCls;
                DoubleMatrix y = new DoubleMatrix(1, AlgConsHSoftmax.nodeSize);
                y.put(nxtNdIdxInCls, 1);
    	        acts.put("y" + t, y);
    	        
    	        DoubleMatrix py = acts.get("py"+t);
    	        DoubleMatrix pc = acts.get("pc"+t);
                cas_logLkHd -= Math.log(py.get(nxtNdIdxInCls))/(infos.size()-1);
                cas_logLkHd -= Math.log(pc.get(nodeCls))/(infos.size()-1);

                DoubleMatrix prob = py.mul(pc.get(nodeCls));
                DoubleMatrix[] otherProb = new DoubleMatrix[AlgConsHSoftmax.cNum-1];
                int cCnt = 0;
                for(int c=0; c<AlgConsHSoftmax.cNum; c++) {
                	if(c==nodeCls) {
                		continue;
                	}
                	DoubleMatrix s = acts.get("s"+t);
                	DoubleMatrix hatYt = s.mmul(output.Wsy[c]).add(output.by[c]);
                    DoubleMatrix predictYt = Activer.softmax(hatYt);
                    otherProb[cCnt] = predictYt.mul(pc.get(c));
                    cCnt++;
                }
                cas_mrr += LossFunction.calcMRR(prob, nxtNdIdxInCls, otherProb)/(infos.size()-1);
                
                prevTm = curTm;
			}
			synchronized(logLkHd) {
				logLkHd += cas_logLkHd;
			}
			synchronized(mrr) {
				mrr -= cas_mrr;
			}
		}
		
		private String consumeMissions() {
    		synchronized(MultiThreadCons.missions) {
    			if(!MultiThreadCons.missions.isEmpty()) {
    				return MultiThreadCons.missions.remove(0);
    			} else {
    				return null;
    			}
    		}
    	}
		
		private void missionOver() {
			
			boolean isCompleted = false;
			while(!isCompleted) {
				synchronized(MultiThreadCons.canRevised) {
					if(MultiThreadCons.canRevised) {
						MultiThreadCons.canRevised = false;
						synchronized(MultiThreadCons.missionOver) {
							MultiThreadCons.missionOver++;
							MultiThreadCons.canRevised = true;
							isCompleted = true;
						}
					}
				}
			}
		}
		
		/* (non-Javadoc)
		 * @see java.lang.Runnable#run()
		 */
		@Override
		public void run() {
			while(!CollectionHelper.isEmpty(MultiThreadCons.missions)) {
				String seq = consumeMissions();
				if(StringHelper.isEmpty(seq)) {
					continue;
				}
				mainProc(seq);
			}
			
			missionOver();
		}
	}
}
