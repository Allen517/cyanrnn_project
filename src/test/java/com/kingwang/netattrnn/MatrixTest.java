/**   
 * @package	com.kingwang.netrnn
 * @File		MatrixTest.java
 * @Crtdate	Dec 8, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn;

import org.jblas.DoubleMatrix;

/**
 *
 * @author King Wang
 * 
 * Dec 8, 2016 4:21:37 PM
 * @version 1.0
 */
public class MatrixTest {

	public static void main(String[] args) {
		
		DoubleMatrix mat1 = new DoubleMatrix(2,2);
		DoubleMatrix vec1 = mat1.getRow(1);
		vec1.put(1, 1);
		mat1.putRow(1, vec1);
		System.out.println(vec1.get(1));
		System.out.println(mat1.get(1, 1));
	}
}
