/**   
 * @package	com.kingwang.ctsrnn.utils
 * @File		BatchDerivative.java
 * @Crtdate	Jul 3, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn.batchderv;

import java.util.Map;

import org.jblas.DoubleMatrix;

/**
 *
 * @author King Wang
 * 
 * Jul 3, 2016 10:17:50 PM
 * @version 1.0
 */
public interface BatchDerivative {

	public void clearBatchDerv();
	
	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac);
	
}
