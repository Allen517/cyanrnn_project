/**   
 * @package	com.kingwang.rnncdm.lstm
 * @File		AlgCons.java
 * @Crtdate	May 25, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.cons;

/**
 *
 * @author King Wang
 * 
 * May 25, 2016 5:09:26 PM
 * @version 1.0
 */
public class AlgCons {

	public static double eps = 10e-8;
	
	public static int epoch = 0;
	public static int validCycle = 0;
	
	public static int stopCount = 0;
	
	public static String rnnType;
	
	public static String trainStrategy;
	//AdaGrad's parameter
	public static double lr = 0;
	//Adam's parameters
	public static double beta1 = .9;
    public static double beta2 = .999;
	
	public static double initScale = 0;
	public static double biasInitVal = 0;
	
	public static int[] rangePos;
	
	public static int cNum = 0;
	
	//code size
	public static int codeSize = 0;
	//in input layer, the size of dynamic input 
	public static int inDynSize = 0;
	//in input layer, the size of fixed input
	public static int inFixedSize = 0;
	public static int hiddenSize = 0;
	public static int nodeSize = 0;
	
	public static double maxObsTm = 0;
	public static double tmDiv = 0;
	
	public static String casFile = "";
	public static String crsValFile = "";
	public static String codeFile = "";
	public static String outFile = "";
	
	//mini-batch related
	public static int minibatchCnt = 0;
	
	//Will be continued from the last training
	public static boolean isContTraining = false;
	public static String lastModelFile = "";
}
