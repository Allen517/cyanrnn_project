/**   
 * @package	utils
 * @File		SortedTuple.java
 * @Crtdate	Mar 2, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn.comm.utils;

import java.util.Arrays;
import java.util.List;
import java.util.Stack;

/**
 *
 * @author King Wang
 * 
 * Mar 2, 2016 2:41:03 PM
 * @version 1.0
 */
public class TVecTools<T extends Comparable<T>> {

	public void quickSortWithIndex(List<T> vec, int[] vecIdx) {
		
		if(vec==null || vecIdx==null) {
			throw new IllegalArgumentException("Vector is error!");
		}
		
		if(vec.size()!=vecIdx.length) {
			throw new IllegalArgumentException("The two vector is not equal in length!");
		}
		
		Stack<Integer> index=new Stack<Integer>();
        int start=0;
        int end=vec.size()-1;
        
        int pivotPos;
                
        index.push(start);
        index.push(end);
                
        while(!index.isEmpty()){
            end=index.pop();
            start=index.pop();
            pivotPos=partition(vec, vecIdx,start,end);
            if(start<pivotPos-1){
                index.push(start);
                index.push(pivotPos-1);
            }
            if(end>pivotPos+1){
                index.push(pivotPos+1);
                index.push(end);
            }
        }
	}
	
	private int partition(List<T> vec, int[] vecIdx,int start,int end){//分块方法，在数组a中，对下标从start到end的数列进行划分
		
        T pivot = vec.get(start); //把比pivot(初始的pivot=a[start]小的数移动到pivot的左边
        int pivotIdx = vecIdx[start];
        
        while(start<end){ //把比pivot大的数移动到pivot的右边
            while(start<end && vec.get(end).compareTo(pivot)>=0) {
            	end--;
            }
            vec.set(start, vec.get(end));
            vecIdx[start]=vecIdx[end];
            while(start<end && vec.get(start).compareTo(pivot)<=0) {
            	start++;
            }
            vec.set(end, vec.get(start));
            vecIdx[end]=vecIdx[start];
        }
        vec.set(start, pivot);
        vecIdx[start]=pivotIdx;
        
        return start;//返回划分后的pivot的位置
        //printArray(a);
	}
	
	public static void main(String[] args) {
		
		TVecTools<Double> tools = new TVecTools<Double>();

		Double[] vecArr = {.5, .2, .8, .3, .4};
		List<Double> vecList = Arrays.asList(vecArr);
		int[] vecIdx = {0,1,2,3,4};
	
		tools.quickSortWithIndex(vecList, vecIdx);
		for(int i=0; i<vecList.size(); i++) {
			System.out.println(vecList.get(i)+","+vecIdx[i]);
		}
	}
}
