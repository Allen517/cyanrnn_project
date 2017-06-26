package com.kingwang.netattrnn.cells.main.hist;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.jblas.DoubleMatrix;

import com.kingwang.netattrnn.batchderv.impl.AttBatchDerivative;
import com.kingwang.netattrnn.batchderv.impl.GRUBatchDerivative;
import com.kingwang.netattrnn.batchderv.impl.InputBatchDerivative;
import com.kingwang.netattrnn.batchderv.impl.OutputBatchWithHSoftmaxDerivative;
import com.kingwang.netattrnn.cells.impl.GRU;
import com.kingwang.netattrnn.cells.impl.InputLayer;
import com.kingwang.netattrnn.cells.impl.OutputLayerWithHSoftMax;
import com.kingwang.netattrnn.cells.impl.hist.Attention_alphaReg;
import com.kingwang.netattrnn.comm.utils.CollectionHelper;
import com.kingwang.netattrnn.comm.utils.Config;
import com.kingwang.netattrnn.comm.utils.FileUtil;
import com.kingwang.netattrnn.comm.utils.StringHelper;
import com.kingwang.netattrnn.cons.AlgConsHSoftmax;
import com.kingwang.netattrnn.cons.MultiThreadCons;
import com.kingwang.netattrnn.dataset.DataLoader;
import com.kingwang.netattrnn.dataset.Node4Code;
import com.kingwang.netattrnn.dataset.SeqLoader;
import com.kingwang.netattrnn.evals.hist.CyanRNNModelEvals_alphaReg;
import com.kingwang.netattrnn.utils.InputEncoder;
import com.kingwang.netattrnn.utils.MatIniter;
import com.kingwang.netattrnn.utils.MatIniter.Type;
import com.kingwang.netattrnn.utils.TmFeatExtractor;

public class CyanRNN_alphaReg {
	private InputLayer input;
	private Attention_alphaReg att;
//	private OutputLayer output;
	private OutputLayerWithHSoftMax output;
    private GRU gru;
    private InputBatchDerivative inputBatchDerv;
    private GRUBatchDerivative rnnBatchDerv;
    private AttBatchDerivative attBatchDerv;
//  private OutputBatchDerivative outputBatchDerv;
    private OutputBatchWithHSoftmaxDerivative outputBatchDerv;
    
    private Double tm_input;
    private Double tm_rnn;
    private Double tm_att;
    private Double tm_output;
    
    private DataLoader casLoader;
    
    public CyanRNN_alphaReg(int nodeSize, int inDynSize, int inFixedSize, int attSize, int hiddenSize 
    				, int cNum, DataLoader casLoader, MatIniter initer) {
    	if(AlgConsHSoftmax.rnnType.equalsIgnoreCase("gru")) {
    		gru = new GRU(inDynSize, inFixedSize, attSize, initer); 
    		rnnBatchDerv = new GRUBatchDerivative();
    	}
    	attBatchDerv = new AttBatchDerivative();
    	outputBatchDerv = new OutputBatchWithHSoftmaxDerivative();
    	inputBatchDerv = new InputBatchDerivative();
        input = new InputLayer(nodeSize, inDynSize, initer);
        att = new Attention_alphaReg(inDynSize, inFixedSize, attSize, hiddenSize, initer);
        output = new OutputLayerWithHSoftMax(inDynSize, inFixedSize, attSize, hiddenSize, cNum, initer);
        if(AlgConsHSoftmax.isContTraining) {
    		gru.loadCellParameter(AlgConsHSoftmax.lastModelFile);
    		att.loadCellParameter(AlgConsHSoftmax.lastModelFile);
    		output.loadCellParameter(AlgConsHSoftmax.lastModelFile);
    		input.loadCellParameter(AlgConsHSoftmax.lastModelFile);
    	}
        this.casLoader = casLoader;
    }
    
    private List<String> getMissions(List<String> sequence) {
    	
    	List<String> missions = new ArrayList<>();
    	for(String seq : sequence) {
    		missions.add(seq);
    	}
    	
    	return missions;
    }
    
    private void calcGradientByMiniBatch(List<String> sequence) {
		
    	MultiThreadCons.missions = getMissions(sequence);
    	MultiThreadCons.missionSize = sequence.size();
    	MultiThreadCons.missionOver = 0;
    	
		ExecutorService exec = Executors.newCachedThreadPool();
		for (int i = 0; i < MultiThreadCons.threadNum; i++) {
			exec.execute(new ForwardExec());
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
    
    private void clearBatchDerv() {
    	outputBatchDerv.clearBatchDerv();
    	attBatchDerv.clearBatchDerv();
    	rnnBatchDerv.clearBatchDerv();
    	inputBatchDerv.clearBatchDerv();
    }
    
    private void train(DataLoader casLoader, String outFile) {
    	
    	OutputStreamWriter oswLog = FileUtil.getOutputStreamWriter("log", true);
    	
    	double minCrsVal = Double.MAX_VALUE;
    	double minCrsValIter = -1;
    	int stopCount = 0;
    	
    	output.writeCellParameter(outFile+".iter0", true);
    	att.writeCellParameter(outFile+".iter0", true);
        gru.writeCellParameter(outFile+".iter0", true);
    	input.writeCellParameter(outFile+".iter0", true);
        for (int epochT = 1; epochT < AlgConsHSoftmax.epoch; epochT++) {
        	double start = System.currentTimeMillis();
        	tm_input = .0;
        	tm_rnn = .0;
        	tm_att = .0;
        	tm_output = .0;
        	MultiThreadCons.epochTrainError = 0;
            List<String> sequence = casLoader.getBatchData(AlgConsHSoftmax.minibatchCnt);
        	calcGradientByMiniBatch(sequence);
        	if(AlgConsHSoftmax.trainStrategy.equalsIgnoreCase("adagrad")) {
        		output.updateParametersByAdaGrad(outputBatchDerv, AlgConsHSoftmax.lr);
        		att.updateParametersByAdaGrad(attBatchDerv, AlgConsHSoftmax.lr);
        		gru.updateParametersByAdaGrad(rnnBatchDerv, AlgConsHSoftmax.lr);
        		input.updateParametersByAdaGrad(inputBatchDerv, AlgConsHSoftmax.lr);
        	}
        	if(AlgConsHSoftmax.trainStrategy.equalsIgnoreCase("adam")) {
        		output.updateParametersByAdam(outputBatchDerv, AlgConsHSoftmax.lr, AlgConsHSoftmax.beta1, AlgConsHSoftmax.beta2, epochT);
        		att.updateParametersByAdam(attBatchDerv, AlgConsHSoftmax.lr, AlgConsHSoftmax.beta1, AlgConsHSoftmax.beta2, epochT);
        		gru.updateParametersByAdam(rnnBatchDerv, AlgConsHSoftmax.lr, AlgConsHSoftmax.beta1, AlgConsHSoftmax.beta2, epochT);
        		input.updateParametersByAdam(inputBatchDerv, AlgConsHSoftmax.lr
        										, AlgConsHSoftmax.beta1, AlgConsHSoftmax.beta2, epochT);
        	}
        	clearBatchDerv();
        	System.out.println("Iter = " + epochT + ", error = " + MultiThreadCons.epochTrainError/sequence.size()  
        			+ ", time = " + (System.currentTimeMillis() - start) / 1000 + "s @input: "+tm_input/1000+"s; rnn: "
        			+ tm_rnn/1000 + "s; att: " + tm_att/1000 + "s; output: " + tm_output/1000 + "s");
        	FileUtil.writeln(oswLog, "Iter = " + epochT + ", error = " + MultiThreadCons.epochTrainError/sequence.size()  
        			+ ", time = " + (System.currentTimeMillis() - start) / 1000 + "s @input: "+tm_input/1000+"s; rnn: "
        			+ tm_rnn/1000 + "s; att: " + tm_att/1000 + "s; output: " + tm_output/1000 + "s");
            if(epochT%AlgConsHSoftmax.validCycle==0) {
//            	double validRes = RNNModelEvals.validationInOtherWay(input, cell, AlgConsHSoftmax.nodeSize
//            						, casLoader.getCrsValSeq(), oswLog);
            	CyanRNNModelEvals_alphaReg rnnEvals = new CyanRNNModelEvals_alphaReg(input, gru, att, output, casLoader, oswLog);
            	double validRes = rnnEvals.validationOnIntegration();
            	if(validRes<minCrsVal) {
            		minCrsVal = validRes;
            		minCrsValIter = epochT;
            		stopCount = 0;
            	} else {
            		stopCount++;
            	}
            	if(stopCount==AlgConsHSoftmax.stopCount) {
            		System.out.println("The best model is located in iter "+minCrsValIter);
                    FileUtil.writeln(oswLog, "The best model is located in iter "+minCrsValIter);
            		break;
            	}
            	output.writeCellParameter(outFile+".iter"+epochT, true);
            	att.writeCellParameter(outFile+".iter"+epochT, true);
            	gru.writeCellParameter(outFile+".iter"+epochT, true);
            	input.writeCellParameter(outFile+".iter"+epochT, true);
            }
        }
    }
    
    class ForwardExec implements Runnable {

    	private void forwardAndBackwardPass(String seq) {
    		
    		Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
            // forward pass
            List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq);
            if(infos.size()<3) { //skip short cascades
            	return;
            }
            String iid = infos.remove(0);
            double prevTm = 0;
            int missCnt = 0;
            double tmInInput=0, tmInGru=0, tmInAtt=0, tmInOutput=0; 
            for (int t=0; t<infos.size()-1; t++) {
            	String[] curInfo = infos.get(t).split(",");
            	String[] nextInfo = infos.get(t+1).split(",");
            	//translating string node to node index in repMatrix
            	String curNd = curInfo[0];
            	String nxtNd = nextInfo[0];
            	double curTm = Double.parseDouble(curInfo[1]);
            	if(!casLoader.getCodeMaps().containsKey(curNd)) {//if curNd isn't located in nodeDict
//            		System.out.println("Missing node"+curNd);
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
            	
            	double st_input = System.currentTimeMillis();
            	input.active(t, acts);
            	double end_input = System.currentTimeMillis();
            	tmInInput += end_input-st_input;
                gru.active(t, acts);
                double end_gru = System.currentTimeMillis();
                tmInGru += end_gru-end_input;
                att.active(t, acts);
                double end_att = System.currentTimeMillis();
                tmInAtt += end_att-end_gru;
                output.active(t, acts, nodeCls);
                double end_output = System.currentTimeMillis();
                tmInOutput += end_output-end_att;
               
                //actual u in class
                DoubleMatrix cls = new DoubleMatrix(1, AlgConsHSoftmax.cNum);
                cls.put(nodeCls, 1);
                acts.put("cls" + t, cls);
                
                int nxtNdIdxInCls = nxtNd4Code.idxInCls;
                DoubleMatrix y = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[nodeCls]);
                y.put(nxtNdIdxInCls, 1);
    	        acts.put("y" + t, y);
    	        
    	        DoubleMatrix py = acts.get("py"+t);
    	        DoubleMatrix pc = acts.get("pc"+t);
    	        DoubleMatrix alpha = acts.get("alpha"+t);
    	        double normAlpha = Math.pow(0.5*AlgConsHSoftmax.lambda_alpha*alpha.norm2(), 2.);
    	        MultiThreadCons.epochTrainError += normAlpha/(infos.size()-1);
    	        MultiThreadCons.epochTrainError -= Math.log(py.get(nxtNdIdxInCls))/(infos.size()-1);
    	        MultiThreadCons.epochTrainError -= Math.log(pc.get(nodeCls))/(infos.size()-1);
//                MultiThreadCons.epochTrainError += (logft-Math.log(py.get(nxtNdIdx)))/(infos.size()-1);
    	        
    	        prevTm = curTm;
        	}
//            System.out.println("###Forwarding### time cost in input layer: "+tmInInput/1000.+"s, @"+iid);
//            System.out.println("###Forwarding### time cost in gru layer: "+tmInGru/1000.+"s, @"+iid);
//            System.out.println("###Forwarding### time cost in att layer: "+tmInAtt/1000.+"s, @"+iid);
//            System.out.println("###Forwarding### time cost in output layer: "+tmInOutput/1000.+"s, @"+iid);
            //backward pass
            double st_output_bptt = System.currentTimeMillis();
            output.bptt(acts, infos.size()-2);
            double end_output_bptt = System.currentTimeMillis();
            synchronized(tm_output) {
            	tm_output += end_output_bptt-st_output_bptt+tmInOutput;
            }
//            System.out.println("###Backwarding### time cost in output layer: "+(end_output_bptt-st_output_bptt)/1000.+"s, @"+iid);
            att.bptt(acts, infos.size()-2, output);
            double end_att_bptt = System.currentTimeMillis();
            synchronized(tm_att) {
            	tm_att += end_att_bptt-end_output_bptt+tmInAtt;
            }
//            System.out.println("###Backwarding### time cost in att layer: "+(end_att_bptt-end_output_bptt)/1000.+"s, @"+iid);
            gru.bptt(acts, infos.size()-2, att);
            double end_gru_bptt = System.currentTimeMillis();
            synchronized(tm_rnn) {
            	tm_rnn += end_gru_bptt-end_att_bptt+tmInGru;
            }
//            System.out.println("###Backwarding### time cost in gru layer: "+(end_gru_bptt-end_att_bptt)/1000.+"s, @"+iid);
            input.bptt(acts, infos.size()-2, output, att, gru);
            double end_input_bptt = System.currentTimeMillis();
            synchronized(tm_input) {
            	tm_input += end_input_bptt-end_gru_bptt+tmInInput;
            }
//            System.out.println("###Backwarding### time cost in input layer: "+(end_input_bptt-end_gru_bptt)/1000.+"s, @"+iid);
            
            synchronized(inputBatchDerv) {
            	inputBatchDerv.batchDervCalc(acts, 1./MultiThreadCons.missionSize);
            }
            synchronized(rnnBatchDerv) {
            	rnnBatchDerv.batchDervCalc(acts, 1./MultiThreadCons.missionSize);
            }
            synchronized(attBatchDerv) {
            	attBatchDerv.batchDervCalc(acts, 1./MultiThreadCons.missionSize);
            }
            synchronized(outputBatchDerv) {
            	outputBatchDerv.batchDervCalc(acts, 1./MultiThreadCons.missionSize);
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
			// TODO Auto-generated method stub
			while(!CollectionHelper.isEmpty(MultiThreadCons.missions)) {
				String seq = consumeMissions();
				if(StringHelper.isEmpty(seq)) {
					continue;
				}
				forwardAndBackwardPass(seq);
			}
			
			missionOver();
		}
    	
    }
    
    public static void main(String[] args) {
    	
    	if(args.length<1) {
    		System.out.println("Please input configuration file");
    		return;
    	}

    	try {
    		Map<String, String> config = Config.getConfParams(args[0]);
    		//Files
    		AlgConsHSoftmax.casFile = config.get("cas_file");
    		AlgConsHSoftmax.crsValFile = config.get("crs_val_file");
    		AlgConsHSoftmax.isContTraining = Boolean.parseBoolean(config.get("is_cont_training"));
    		if(AlgConsHSoftmax.isContTraining) {
    			AlgConsHSoftmax.lastModelFile = config.get("last_rnn_model");
    		} 
			AlgConsHSoftmax.freqFile = config.get("freq_file");
    		AlgConsHSoftmax.outFile = config.get("out_file");
    		AlgConsHSoftmax.rnnType = config.get("rnn_type");
    		AlgConsHSoftmax.trainStrategy = config.get("train_strategy");
    		//Learning Parameters
    		AlgConsHSoftmax.lr = Double.parseDouble(config.get("lr"));
    		if(AlgConsHSoftmax.trainStrategy.equalsIgnoreCase("adam")) {
    			AlgConsHSoftmax.lr = Double.parseDouble(config.get("lr"));
    			AlgConsHSoftmax.beta1 = Double.parseDouble(config.get("beta1"));
    			AlgConsHSoftmax.beta2 = Double.parseDouble(config.get("beta2"));
    		}
    		AlgConsHSoftmax.tmDiv = Double.parseDouble(config.get("time_div"));
    		AlgConsHSoftmax.windowSize = Integer.parseInt(config.get("window_size"));
    		//Model Parameters
    		AlgConsHSoftmax.lambda_alpha = Double.parseDouble(config.get("lambda_alpha"));
    		AlgConsHSoftmax.initScale = Double.parseDouble(config.get("init_scale"));
    		AlgConsHSoftmax.biasInitVal = Double.parseDouble(config.get("bias_init_val"));
    		AlgConsHSoftmax.stopCount = Integer.parseInt(config.get("stop_count"));
    		AlgConsHSoftmax.cNum = Integer.parseInt(config.get("class_num"));
    		AlgConsHSoftmax.nodeSize = Integer.parseInt(config.get("node_size"));
    		AlgConsHSoftmax.inFixedSize = Integer.parseInt(config.get("in_fixed_size"));
    		AlgConsHSoftmax.inDynSize = Integer.parseInt(config.get("in_dyn_size"));
    		AlgConsHSoftmax.hiddenSize = Integer.parseInt(config.get("hidden_size"));
    		AlgConsHSoftmax.attSize = Integer.parseInt(config.get("att_size"));
    		AlgConsHSoftmax.epoch = Integer.parseInt(config.get("epoch"));
    		AlgConsHSoftmax.validCycle = Integer.parseInt(config.get("validation_cycle"));
    		AlgConsHSoftmax.minibatchCnt = Integer.parseInt(config.get("no_of_minibatch_values"));
    		//System settings
    		MultiThreadCons.threadNum = Integer.parseInt(config.get("thread_num"));
    		MultiThreadCons.sleepSec = Double.parseDouble(config.get("sleep_sec"));
    		
    		Config.printConf(config, "log");
    	} catch(IOException e) {}
    	
        DataLoader cl = new DataLoader(AlgConsHSoftmax.casFile, AlgConsHSoftmax.crsValFile
        			, AlgConsHSoftmax.freqFile, AlgConsHSoftmax.cNum);
        CyanRNN_alphaReg rnn = new CyanRNN_alphaReg(AlgConsHSoftmax.nodeSize, AlgConsHSoftmax.inDynSize, AlgConsHSoftmax.inFixedSize
        					, AlgConsHSoftmax.attSize, AlgConsHSoftmax.hiddenSize, AlgConsHSoftmax.cNum
        					, cl, new MatIniter(Type.SVD));
        rnn.train(cl, AlgConsHSoftmax.outFile);
    }

}
