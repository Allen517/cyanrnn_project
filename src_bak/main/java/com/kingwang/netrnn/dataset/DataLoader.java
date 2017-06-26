package com.kingwang.netrnn.dataset;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.jblas.DoubleMatrix;

import com.kingwang.netrnn.comm.utils.FileUtil;
import com.kingwang.netrnn.comm.utils.StringHelper;
import com.kingwang.netrnn.cons.AlgConsHSoftmax;

public class DataLoader {

    private Map<String, Node4Code> codeMaps = new HashMap<>();
    private List<String> sequence = new ArrayList<String>();
    private DoubleMatrix repMat;
    private List<String> crsValSeq = new ArrayList<>();
    private int clsNum = 0;
    
    public DataLoader() {}
    
    public DataLoader(String dataFile) {
    	sequence = loadMemeFormatData(dataFile, true);
    }
    
    public DataLoader(String dataFile, int nodeSize, int repDim) {
    	sequence = loadMemeFormatData(dataFile, true);
    	codeMaps = new HashMap<>();
    }
    
    public DataLoader(String dataFile, String crsValFile, String repFile, String freqFile, int clsNum) {
		if(StringHelper.isEmpty(freqFile)) {
			codeMaps = new HashMap<>();
		} else {
			codeMaps = setCodeMaps(freqFile, clsNum);
		}
		setRepMat(repFile);
		sequence = loadMemeFormatData(dataFile, true);
		crsValSeq = loadMemeFormatData(crsValFile, true);
	}
    
    public DataLoader(String dataFile, String crsValFile, String freqFile, int clsNum) {
		if(StringHelper.isEmpty(freqFile)) {
			codeMaps = new HashMap<>();
		} else {
			codeMaps = setCodeMaps(freqFile, clsNum);
		}
		sequence = loadMemeFormatData(dataFile, true);
		crsValSeq = loadMemeFormatData(crsValFile, true);
	}
    
    private void setRepMat(String repFile) {
    	repMat = new DoubleMatrix(AlgConsHSoftmax.nodeSize, AlgConsHSoftmax.inRepSize);
    	try(BufferedReader br = FileUtil.getBufferReader(repFile)) {
    		String line = null;
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(" ");
    			String ndId = elems[0];
    			DoubleMatrix lnMat = new DoubleMatrix(1, AlgConsHSoftmax.inRepSize);
    			for(int i=1; i<elems.length; i++) {
    				lnMat.put(i-1, Double.parseDouble(elems[i]));
    			}
    			repMat.putRow(Integer.parseInt(elems[0]), lnMat);
    		}
    	} catch(IOException e) {
    		e.printStackTrace();
    	}
    }
    
    /**
     * Simplest dictionary where the reindices are arranged by locations
     * 
     * @param dictFile
     * @return
     */
    private Map<String, Node4Code> setCodeMaps(String freqFile, int clsNum) {
    	
    	Map<String, Node4Code> codeMaps = new HashMap<>();
    	
    	try(BufferedReader br = FileUtil.getBufferReader(freqFile)) {
    		String line = null;
    		AlgConsHSoftmax.nodeSizeInCls = new int[clsNum];
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(",");
    			if(elems.length<2) {
    				continue;
    			}
    			//setting node code
    			String nodeId = elems[0];
    			double freqDist = Double.parseDouble(elems[2]);
    			//set class index according to freqDist
    			int nodeCls = (int) Math.floor((freqDist-AlgConsHSoftmax.eps)*AlgConsHSoftmax.cNum);
    			//set index in class
    			int idxInCls = AlgConsHSoftmax.nodeSizeInCls[nodeCls];
    			Node4Code nCode = new Node4Code(freqDist, nodeCls, idxInCls);
    			
    			codeMaps.put(nodeId, nCode);
    			AlgConsHSoftmax.nodeSizeInCls[nodeCls]++;
    		}
    	} catch(IOException e) {
    		
    	}
    	
    	return codeMaps;
    }
    
    private List<List<String>> loadMemeFormatDataInBatch(String filePath, int minibatchCnt) {
    	
    	List<List<String>> seqBatch = new ArrayList<>();
    	
    	Map<Integer, List<String>> seqBatchMap = new HashMap<>();
    	int maxCasLen = -1;
//    	System.out.println("***********TEST***********");//for test
//    	Random rand = new Random();//for test
    	try(BufferedReader br = FileUtil.getBufferReader(filePath)) {
    		String line = null;
    		while((line=br.readLine())!=null) {
//    			if(rand.nextDouble()<.95) { //for test
//    				continue;
//    			}
    			String[] elems = line.split(",");
    			if(elems.length<3) {
    				continue;
    			}
    			int casLen = (elems.length-1)/2;
    			List<String> sequence = null;
    			if(seqBatchMap.isEmpty() || !seqBatchMap.containsKey(casLen)) {
    				sequence = new ArrayList<>();
    			} else {
    				sequence = seqBatchMap.get(casLen);
    			}
    			sequence.add(line);
    			seqBatchMap.put(casLen, sequence);
    			if(maxCasLen<casLen) {
    				maxCasLen = casLen;
    			}
    		}
    	} catch(IOException e) {
    		
    	}
    	
    	List<String> oneBatchSeq = new ArrayList<>();
    	for(int len=1; len<maxCasLen; len++) {
    		if(!seqBatchMap.containsKey(len)) {
    			continue;
    		}
    		List<String> sequence = seqBatchMap.get(len);
    		for(String seq : sequence) {
    			if(oneBatchSeq.size()<minibatchCnt) {
    				oneBatchSeq.add(seq);
    			} else {
    				seqBatch.add(oneBatchSeq);
    				oneBatchSeq = new ArrayList<>();
    				oneBatchSeq.add(seq);
    			}
    		}
    	}
    	seqBatch.add(oneBatchSeq);
    	
    	return seqBatch;
    }
    
    private List<String> loadMemeFormatData(String filePath, boolean initDict) {
    	
    	if(StringHelper.isEmpty(filePath)) {
    		return Collections.emptyList();
    	}
    	
    	List<String> seq = new ArrayList<>();
    	
    	try(BufferedReader br = FileUtil.getBufferReader(filePath)) {
    		String line = null;
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(",");
    			if(elems.length<3) {
    				continue;
    			}
    			//TODO
    			if(elems.length>2*1000) {
					continue;
				}
    			seq.add(line);
    		}
    	} catch(IOException e) {
    		
    	}
    	
    	return seq;
    }
    
	/**
	 * @return the sequence
	 */
	public List<String> getSequence() {
		return sequence;
	}

	/**
	 * @param sequence the sequence to set
	 */
	public void setSequence(List<String> sequence) {
		this.sequence = sequence;
	}
	
	public List<String> getCrsValSeq() {
		return crsValSeq;
	}

	public void setCrsValSeq(List<String> crsValSeq) {
		this.crsValSeq = crsValSeq;
	}

	/**
	 * @return the seqBatch
	 */
	public List<String> getBatchData(int miniBathCnt) {
		
		List<String> batchData = new ArrayList<>();
		Random rand = new Random();
		for(int i=0; i<miniBathCnt; i++) {
			int idx = rand.nextInt(sequence.size());
			batchData.add(sequence.get(idx));
		}
		
		return batchData;
	}

	/**
	 * @return the codeMaps
	 */
	public Map<String, Node4Code> getCodeMaps() {
		return codeMaps;
	}

	/**
	 * @return the repMat
	 */
	public DoubleMatrix getRepMat() {
		return repMat;
	}

	/**
	 * @param repMat the repMat to set
	 */
	public void setRepMat(DoubleMatrix repMat) {
		this.repMat = repMat;
	}

}
