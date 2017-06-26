package com.kingwang.netrnn.utils;

import java.util.Random;

import org.jblas.DoubleMatrix;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class MatIniter {
    private static Random random = new Random();

    public enum Type {
        Uniform, Gaussian, SVD, Test
    }
    
    private Type type;
    private double scale = 0.01;
    private double miu = 0;
    private double sigma = 0.01;

    public MatIniter(Type type) {
        this.type = type;
    }
    
    public MatIniter(Type type, double scale, double miu, double sigma) {
        this.type = type;
        this.scale = scale;
        this.miu = miu;
        this.sigma = sigma;
    }
    
    public DoubleMatrix uniform(int rows, int cols) {
        return DoubleMatrix.rand(rows, cols).mul(2 * scale).sub(scale);
    }
    
    public DoubleMatrix gaussian(int rows, int cols) {
        DoubleMatrix m = new DoubleMatrix(rows, cols);
        for (int i = 0; i < m.length; i++) {
            m.put(i, random.nextGaussian() * sigma + miu);
        }
        return m;
    }
    
    public DoubleMatrix svd(int rows, int cols) {
    	DoubleMatrix m = new DoubleMatrix(rows, cols);
    	
    	Matrix mat = null;
    	if(rows<cols) {
    		mat = Matrix.random(cols, rows);
    	} else {
    		mat = Matrix.random(rows, cols);
    	}
    	SingularValueDecomposition svd = mat.svd();
    	Matrix U = svd.getU();
    	if(rows>cols) {
    		for(int i=0; i<rows; i++) {
    			for(int j=0; j<cols; j++) {
    				m.put(i, j, U.get(i, j));
    			}
    		}
    	} else {
    		for(int i=0; i<rows; i++) {
    			for(int j=0; j<cols; j++) {
    				m.put(i, j, U.get(j, i));
    			}
    		}
    	}
    	
    	return m;
    }

    public Type getType() {
        return type;
    }

    public double getScale() {
        return scale;
    }

    public double getMiu() {
        return miu;
    }

    public double getSigma() {
        return sigma;
    }
    
    public static void main(String[] args) {
    	
    	MatIniter init = new MatIniter(Type.SVD, 0, 0, 0);
    	DoubleMatrix m = init.svd(3, 8);
    	System.out.println(m.rows+","+m.columns);
    	for(int i=0; i<m.rows; i++) {
    		for(int j=0; j<m.columns; j++) {
    			System.out.print(m.get(i,j)+" ");
    		}
    		System.out.println("");
    	}
    }
}