/**   
 * @package	anaylsis.data.util
 * @File		VecTools.java
 * @Crtdate	Aug 10, 2013
 *
 * Copyright (c) 2013 by <a href="mailto:wangyongqing.casia@gmail.com">Allen Wang</a>.   
 */
package com.kingwang.netattrnn.comm.utils;


/**
 *
 * @author Allen Wang
 * 
 * Aug 10, 2013 8:36:16 PM
 * @version 1.0
 */
public class VecTools {

	public static Double[] getFixValVec(Integer size, Double fixVal) {
		
		Double[] vec = new Double[size];
		
		for(int i=0; i<size; i++) {
			vec[i] = fixVal;
		}
		
		return vec;
	}
	
	public static Double[] doubleVecInit(Integer ftuLen) {
		
		Double[] vec = new Double[ftuLen];
		
		for(int i=0; i<ftuLen; i++) {
			vec[i] = .0;
		}
		
		return vec;
	}
	
	public static Double[] vecAdd(Double[] vec1, Double[] vec2) {
		
		if(vec1==null || vec2==null || vec1.length!=vec2.length) {
			throw new IllegalArgumentException("Vector is error!");
		}
		int len = vec1.length;
		Double[] sum = new Double[len];
		for(int i=0; i<len; i++) {
			sum[i] = vec1[i]+vec2[i];
		}
		
		return sum;
	}
	
	public static double[] vecAdd(double[] vec1, double[] vec2) {
		
		if(vec1==null || vec2==null || vec1.length!=vec2.length) {
			throw new IllegalArgumentException("Vector is error!");
		}
		int len = vec1.length;
		double[] sum = new double[len];
		for(int i=0; i<len; i++) {
			sum[i] = vec1[i]+vec2[i];
		}
		
		return sum;
	}
	
	public static double vecMultiply(Double[] vec1, Double[] vec2) {
		
		if(vec1==null || vec2==null || vec1.length!=vec2.length) {
			throw new IllegalArgumentException("Vector is error!");
		}
		double val = .0;
		Integer len = vec1.length;
		for(int i=0; i<len; i++) {
			val += vec1[i]*vec2[i];
		}
		
		return val;
	}
	
	public static double vecMultiply(double[] vec1, double[] vec2) {
		
		if(vec1==null || vec2==null || vec1.length!=vec2.length) {
			throw new IllegalArgumentException("Vector is error!");
		}
		double val = .0;
		Integer len = vec1.length;
		for(int i=0; i<len; i++) {
			val += vec1[i]*vec2[i];
		}
		
		return val;
	}
	
	public static Double[] dotMultiply(Double val, Double[] vec) {
		
		if(vec==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		int len = vec.length;
		Double[] dotMult = new Double[len];
		for(int i=0; i<len; i++) {
			dotMult[i] = val*vec[i];
		}
		
		return dotMult;
	}
	
	public static double[] dotMultiply(double val, double[] vec) {
		
		if(vec==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		int len = vec.length;
		double[] dotMult = new double[len];
		for(int i=0; i<len; i++) {
			dotMult[i] = val*vec[i];
		}
		
		return dotMult;
	}
	
	public static Double[] negVec(Double[] vec) {
		
		if(vec==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		Integer len = vec.length;
		Double[] negVec = new Double[len];
		for(int i=0; i<len; i++) {
			negVec[i] = -vec[i];
		}
		
		return negVec;
	}
	
	public static double[] negVec(double[] vec) {
		
		if(vec==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		int len = vec.length;
		double[] negVec = new double[len];
		for(int i=0; i<len; i++) {
			negVec[i] = -vec[i];
		}
		
		return negVec;
	}
	
	public static Double[] copyVec(Double[] vec) {
		
		if(vec==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		Integer len = vec.length;
		Double[] copyVec = new Double[len];
		for(int i=0; i<len; i++) {
			copyVec[i] = vec[i];
		}
		
		return copyVec;
	}
	
	public static double[] vecAdd(double[] vec1, double[] vec2, double lowBound, double upBound) {
		
		if(vec1==null || vec2==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		if(vec1.length!=vec2.length) {
			throw new IllegalArgumentException("The two vector is not equal in length!");
		}
		
		double[] vecRes = new double[vec1.length];
		for(int i=0; i<vec1.length; i++) {
			vecRes[i] = vec1[i]+vec2[i];
			if(vecRes[i]<lowBound) {
				vecRes[i] = lowBound;
			}
			if(vecRes[i]>upBound) {
				vecRes[i] = upBound;
			}
		}
		
		return vecRes;
	}
	
	public static double[] vecAdd(double[] vec1, double[] vec2, double bound, Boolean isBigger) {
		
		if(vec1==null || vec2==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		if(vec1.length!=vec2.length) {
			throw new IllegalArgumentException("The two vector is not equal in length!");
		}
		
		double[] vecRes = new double[vec1.length];
		for(int i=0; i<vec1.length; i++) {
			vecRes[i] = vec1[i]+vec2[i];
			if(isBigger && vecRes[i]<bound) {
				vecRes[i] = bound;
			}
			if(!isBigger && vecRes[i]>bound) {
				vecRes[i] = bound;
			}
		}
		
		return vecRes;
	}
	
	public static double[] vecMinus(double[] vec1, double[] vec2) {
		
		return VecTools.vecAdd(vec1, VecTools.negVec(vec2));
	}
	
	public static double getVecSum(double[] vec) {
		
		if(vec==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		double retVal = .0;
		for(int i=0; i<vec.length; i++) {
			retVal += vec[i];
		}
		
		return retVal;
	}
	
	public static int getVecSum(int[] vec) {
		
		if(vec==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		int retVal = 0;
		for(int i=0; i<vec.length; i++) {
			retVal += vec[i];
		}
		
		return retVal;
	}
	
	public static double generalDistance(double[] vec1, double[] vec2, double typeVal) {
		
		if(vec1==null || vec2==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		if(vec1.length!=vec2.length) {
			throw new IllegalArgumentException("The two vector is not equal in length!");
		}
		
		double dist = .0;
		for(int i=0; i<vec1.length; i++) {
			dist += Math.pow(vec1[i]-vec2[i], typeVal);
		}
		
		return dist;
	}
	
}
