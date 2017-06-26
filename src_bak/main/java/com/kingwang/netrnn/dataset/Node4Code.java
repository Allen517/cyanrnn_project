/**   
 * @package	com.kingwang.cdmrnn.dataset
 * @File		NodeCode.java
 * @Crtdate	Oct 24, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn.dataset;


/**
 *
 * @author King Wang
 * 
 * Oct 24, 2016 11:35:30 AM
 * @version 1.0
 */
public class Node4Code {

	public int nodeCls;		//the index of class where the code is
	public double freqDist;	//the frequency of node appearance
	public int idxInCls;			//the index of node in class
	
	public Node4Code(double freq, int nodeCls, int idxInCls) {
		this.freqDist = freq;
		this.nodeCls = nodeCls;
		this.idxInCls = idxInCls;
	}
}
