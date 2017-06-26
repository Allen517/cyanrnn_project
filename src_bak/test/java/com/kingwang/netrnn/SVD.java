/**   
 * @package	com.kingwang.ctsrnn
 * @File		SVD.java
 * @Crtdate	Jul 13, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

/**
 *
 * @author King Wang
 * 
 * Jul 13, 2016 10:57:45 AM
 * @version 1.0
 */
public class SVD {

	public static void main(String[] args) {
        // create M-by-N matrix that doesn't have full rank
         int M = 3, N = 8;
         Matrix B = Matrix.random(8, 3);
         Matrix A = Matrix.random(M, N).times(B).times(B.transpose());
         System.out.print("A = ");
         A.print(9, 6);

         // compute the singular vallue decomposition
         System.out.println("A = U S V^T");
         System.out.println();
         SingularValueDecomposition s = A.svd();
         Matrix U = s.getU();
         Matrix V = s.getV();
         Matrix S = s.getS();
         System.out.print("U = ");
         U.print(4, 6);
         System.out.print("Sigma = ");
         S.print(4, 6);
         System.out.print("V = ");
         V.print(4, 6);
         System.out.println("rank = " + s.rank());
         System.out.println("condition number = " + s.cond());
         System.out.println("2-norm = " + s.norm2());

         // print out singular values
         System.out.print("singular values = ");
         Matrix svalues = new Matrix(s.getSingularValues(), 1);
         svalues.print(9, 6);
   }
}
