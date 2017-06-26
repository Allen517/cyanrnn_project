/**   
 * @package	com.kingwang.rnncdm.lstm
 * @File		MultiThreadCons.java
 * @Crtdate	Jun 17, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.cons;

import java.util.List;

/**
 *
 * @author King Wang
 * 
 * Jun 17, 2016 10:39:47 PM
 * @version 1.0
 */
public class MultiThreadCons {

	public static int threadNum;
	public static double sleepSec;
	
    public static double epochTrainError;
    
    public static List<String> missions;
    public static int missionSize;
    
    public static Integer missionOver;
    public static Boolean canRevised=true;
}
