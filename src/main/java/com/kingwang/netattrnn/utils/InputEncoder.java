/**   
 * @package	com.kingwang.cdmrnn.utils
 * @File		InputEncoder.java
 * @Crtdate	Oct 28, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.utils;

import org.jblas.DoubleMatrix;

import com.kingwang.netattrnn.comm.utils.StringHelper;

/**
 *
 * @author King Wang
 * 
 * Oct 28, 2016 9:41:49 AM
 * @version 1.0
 */
public class InputEncoder {

	/**
	 * 
	 * 
	 * @param rangePos the start position records in code(DoubleMatrix). In rangePos, rangePos[0] refers
	 *  to one hot representation; rangePos[1] refers to single representation; range[2:] refers to binary
	 *  representation(which should be changed in future).
	 * @param node
	 * @param codeMaps mapping records for node to Node4Code.
	 * @param code the transferred code for input.
	 * @return code
	 */
	public static DoubleMatrix setBinaryCode(String node, DoubleMatrix code) throws Exception {
    	
    	if(StringHelper.isEmpty(node) || code==null || code.isEmpty()) {
    		throw new Exception("*************WARNING: the node or the code is empty*************");
    	}
    	
    	if(!node.equalsIgnoreCase("null")) {
			code.put(Integer.parseInt(node), 1);
    	} else {
    		throw new Exception("*************WARNING: the node cannot be transferred into " +
					"binary code***************");
    	}
    	
    	return code;
    }
    
    private static void loadOneMat(DoubleMatrix fixedFeat, DoubleMatrix mat, int l) throws Exception {

    	if(l+mat.length>fixedFeat.length) {
    		throw new Exception("***********ERROR: the inputs of fixed feature is out of range************");
    	}
    	
//    	String wrtLn = "loadOneMat"+mat.length+":";
    	for(int k=0; k<mat.length; k++) {
    		fixedFeat.put(l+k, mat.get(k));
//    		wrtLn += fixedFeat.get(l+k)+",";
    	}
//    	FileUtil.writeln(runLog, wrtLn);
    }
    
    /**
     * 
     * 
     * @param t
     * @param curTm
     * @param tmGap
     * @param featLen
     * @param loadMats
     * @return
     * @throws Exception
     */
    public static DoubleMatrix setFixedFeat(int t, int featLen, DoubleMatrix... loadMats) throws Exception {
    	
    	DoubleMatrix fixedFeat = new DoubleMatrix(1, featLen);
    	
    	int l = 0;
    	if(loadMats!=null && loadMats.length>0) {
//    		System.out.println("mat length:"+loadMats.length);
    		for(int i=0; i<loadMats.length; i++) {
    			loadOneMat(fixedFeat, loadMats[i], l);
    			l += loadMats[i].length;
    		}
    	}
    	
    	return fixedFeat;
    }
}
