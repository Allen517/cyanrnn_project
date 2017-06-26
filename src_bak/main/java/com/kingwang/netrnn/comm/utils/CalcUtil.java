/**   
 * @package	com.seocnnet.venus.core.util
 * @File		CalcUtil.java
 * @Crtdate	Jun 8, 2012
 *
 * Copyright (c) 2012 by <a href="mailto:wangyongqing.casia@gmail.com">Allen Wang</a>.   
 */
package com.kingwang.netrnn.comm.utils;

import java.util.List;

/**
 * 
 * @author Allen Wang
 * 
 *         Jun 8, 2012 9:41:07 AM
 * @version 1.0
 */
public class CalcUtil {

	public static double findMaxElem(double[] vals) {
		
		if(vals==null || vals.length<1) {
			return .0;
		}
		
		double maxVal = -Double.MAX_VALUE;
		
		for(int i=0; i<vals.length; i++) {
			if(vals[i]>maxVal) {
				maxVal = vals[i];
			}
		}
		
		return maxVal;
	}
	
	public static int getCeilInt(int dividend, int divisor) {
		if (divisor != 0) {
			return (double) dividend / (double) divisor > (dividend / divisor) ? (dividend / divisor) + 1
					: (dividend / divisor);
		}

		return 0;
	}
	
	public static double getMean(List<Double> values) {
		
		if(CollectionHelper.isEmpty(values)) {
			return .0;
		}
		
		double v = .0;
		for(Double val : values) {
			v += val;
		}
		
		return v/values.size();
	}
	
	public static double getSampleVar(List<Double> values, Double mean) {
		
		if(CollectionHelper.isEmpty(values) || mean==null) {
			return .0;
		}
		
		if(values.size()==1) {
			return .0;
		}

		double v = .0;
		for(Double val : values) {
			v += Math.pow(val-mean, 2.); 
		}
		
		return v/(values.size()-1);
	}
	
	public static double sum(Double[] vals) {
		
		if(vals==null || vals.length<1) {
			return .0;
		}
		
		double sum = .0;
		
		int len = vals.length;
		for(int i=0; i<len; i++) {
			sum += vals[i];
		}
		
		return sum;
	}
	
	public static double sum(double[] vals) {
		
		if(vals==null || vals.length<1) {
			return .0;
		}
		
		double sum = .0;
		
		int len = vals.length;
		for(int i=0; i<len; i++) {
			sum += vals[i];
		}
		
		return sum;
	}
	
	public static Double[] quasiNormalization(Double[] vals) {
		
		if(vals.length<1) {
			return null;
		}
		
		int len = vals.length;
		
		Double[] normVals = new Double[len];
		double minVal = Double.MAX_VALUE;
		for(int i=0; i<len; i++) {
			if(vals[i]<minVal && vals[i]>0) {
				minVal = vals[i];
			}
		}
		
		for(int i=0; i<len; i++) {
			normVals[i] = vals[i]/minVal;
		}
		
		return normVals;
	}
}
