/**   
 * @package	com.kingwang.rnncdm.evals
 * @File		RNNModelMRREvals.java
 * @Crtdate	May 22, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.evals.hist;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;

import com.kingwang.netattrnn.cells.impl.GRU;
import com.kingwang.netattrnn.cells.impl.InputLayer;
import com.kingwang.netattrnn.cells.impl.OutputLayer;
import com.kingwang.netattrnn.cells.impl.hist.Attention_alphaReg;
import com.kingwang.netattrnn.comm.utils.Config;
import com.kingwang.netattrnn.comm.utils.FileUtil;
import com.kingwang.netattrnn.cons.AlgConsHSoftmax;
import com.kingwang.netattrnn.dataset.DataLoader;
import com.kingwang.netattrnn.dataset.Node4Code;
import com.kingwang.netattrnn.dataset.SeqLoader;
import com.kingwang.netattrnn.utils.InputEncoder;
import com.kingwang.netattrnn.utils.LossFunction;
import com.kingwang.netattrnn.utils.MatIniter;
import com.kingwang.netattrnn.utils.MatIniter.Type;
import com.kingwang.netattrnn.utils.TmFeatExtractor;

/**
 *
 * @author King Wang
 * 
 * May 22, 2016 5:03:33 PM
 * @version 1.0
 */
public class Evals {
	
	static InputLayer input;
    static GRU gru;
    static Attention_alphaReg att;
    static OutputLayer output;
	
    @Deprecated
	public static void testOnMRR(InputLayer input, GRU gru, Attention_alphaReg att, OutputLayer output
							, int nodeSize, DataLoader dataLoader, OutputStreamWriter oswLog) {

		List<String> crsValSeq = dataLoader.getCrsValSeq();
		double mrr = .0;
		for (String seq : crsValSeq) {
			Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
			List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq);
            if(infos.size()<3) { //skip short cascades
            	return;
            }
            String iid = infos.remove(0);
			double cas_mrr = 0;
			double prevTm = 0;
			String wrtLn = iid+",";
			int missCnt = 0;
			for (int t = 0; t < infos.size() - 1; t++) {
				String[] curInfo = infos.get(t).split(",");
				String[] nextInfo = infos.get(t + 1).split(",");
				// translating string node to node index in repMatrix
				String curNd = curInfo[0];
            	String nxtNd = nextInfo[0];
            	double curTm = Double.parseDouble(curInfo[1]);
            	if(!dataLoader.getCodeMaps().containsKey(curNd)) {//if curNd isn't located in nodeDict
//            		System.out.println("Missing node"+curNd);
            		missCnt++;
            		curNd = "null";
            		break;//TODO: how to solve "null" node
            	}
            	if(!dataLoader.getCodeMaps().containsKey(nxtNd)) {//if curNd isn't located in nodeDict
            		curNd = "null";
            		break;//TODO: how to solve "null" node
            	}
            	Node4Code nxtNd4Code = dataLoader.getCodeMaps().get(nxtNd);
            	//Set DoubleMatrix code & fixedFeat. It should be a code setter function here.
            	DoubleMatrix tmFeat = TmFeatExtractor.timeFeatExtractor(curTm, prevTm);
            	DoubleMatrix fixedFeat;
				try {
					fixedFeat = InputEncoder.setFixedFeat(t, AlgConsHSoftmax.inFixedSize, tmFeat);
					acts.put("fixedFeat"+t, fixedFeat);
					DoubleMatrix code = new DoubleMatrix(1, AlgConsHSoftmax.inDynSize); 
					code = InputEncoder.setBinaryCode(curNd, code);
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

                //actual u in class
                DoubleMatrix cls = new DoubleMatrix(1, AlgConsHSoftmax.cNum);
                cls.put(nodeCls, 1);
                acts.put("cls" + t, cls);
                
                //p(u|H_i)
                int nxtNdIdxInCls = nxtNd4Code.nodeCls;
                DoubleMatrix py = acts.get("py"+t);
                DoubleMatrix pc = acts.get("pc"+t);
                double cur_mrr = LossFunction.calcMRR(py, nxtNdIdxInCls)/(infos.size()-1);
                wrtLn += nxtNd+","+cur_mrr+",";
                cas_mrr += cur_mrr;
                
                prevTm = curTm;
			}
			mrr += cas_mrr;
			FileUtil.writeln(oswLog, wrtLn.substring(0, wrtLn.length()-1));
		}
		mrr /= crsValSeq.size();
		System.out.println("The MRR of node prediction in Validation: " + mrr);
		FileUtil.writeln(oswLog, "The MRR of node prediction in Validation: " + mrr);

	}
	
	public static void main(String[] args) {
		
		if(args.length<1) {
    		System.out.println("Please input configuration file");
    		return;
    	}

    	try {
    		Map<String, String> config = Config.getConfParams(args[0]);
    		AlgConsHSoftmax.casFile = config.get("cas_file");
    		AlgConsHSoftmax.freqFile = config.get("code_file");
    		AlgConsHSoftmax.outFile = config.get("out_file");
    		AlgConsHSoftmax.rnnType = config.get("rnn_type");
    		AlgConsHSoftmax.lastModelFile = config.get("last_rnn_model");
    		AlgConsHSoftmax.tmDiv = Double.parseDouble(config.get("time_div"));
    		AlgConsHSoftmax.cNum = Integer.parseInt(config.get("class_num"));
    		AlgConsHSoftmax.nodeSize = Integer.parseInt(config.get("code_size"));
    		AlgConsHSoftmax.inFixedSize = Integer.parseInt(config.get("in_fixed_size"));
    		AlgConsHSoftmax.inDynSize = Integer.parseInt(config.get("in_dyn_size"));
    		AlgConsHSoftmax.hiddenSize = Integer.parseInt(config.get("hidden_size"));
    		AlgConsHSoftmax.nodeSize = Integer.parseInt(config.get("node_size"));
    		
    		Config.printConf(config, "log");
    	} catch(IOException e) {}
    	
    	DataLoader dl = new DataLoader(AlgConsHSoftmax.casFile, AlgConsHSoftmax.crsValFile
    			, AlgConsHSoftmax.freqFile, AlgConsHSoftmax.cNum);
    	MatIniter initer = new MatIniter(Type.SVD);
    	if(AlgConsHSoftmax.rnnType.equalsIgnoreCase("gru")) {
    		gru = new GRU(AlgConsHSoftmax.inDynSize, AlgConsHSoftmax.inFixedSize, AlgConsHSoftmax.hiddenSize, initer); 
    	}
        input = new InputLayer(AlgConsHSoftmax.nodeSize, AlgConsHSoftmax.inDynSize, initer);
        att = new Attention_alphaReg(AlgConsHSoftmax.hiddenSize, initer);
        output = new OutputLayer(AlgConsHSoftmax.hiddenSize, AlgConsHSoftmax.nodeSize, initer);
        if(AlgConsHSoftmax.isContTraining) {
    		gru.loadCellParameter(AlgConsHSoftmax.lastModelFile);
    		att.loadCellParameter(AlgConsHSoftmax.lastModelFile);
    		output.loadCellParameter(AlgConsHSoftmax.lastModelFile);
    		input.loadCellParameter(AlgConsHSoftmax.lastModelFile);
    	} 
    	
        OutputStreamWriter osw = FileUtil.getOutputStreamWriter(AlgConsHSoftmax.outFile);
        testOnMRR(input, gru, att, output, AlgConsHSoftmax.nodeSize, dl, osw);
	}
}
