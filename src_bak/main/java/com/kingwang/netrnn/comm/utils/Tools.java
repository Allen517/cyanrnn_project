/**   
 * @package	anaylsis.data.util
 * @File		Tools.java
 * @Crtdate	Aug 4, 2013
 *
 * Copyright (c) 2013 by <a href="mailto:wangyongqing.casia@gmail.com">Allen Wang</a>.   
 */
package com.kingwang.netrnn.comm.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Allen Wang
 * 
 * Aug 4, 2013 11:20:14 AM
 * @version 1.0
 */
public class Tools {

	public static Map<Integer, List<Integer>> getNodeMap(String filePath, String regex) throws IOException {
		
		Map<Integer, List<Integer>> nodeMap	= new HashMap<Integer, List<Integer>>();
		
		BufferedReader br = FileUtil.getBufferReader(filePath);
		String line = null;
		while((line=br.readLine())!=null) {
			String[] elems = line.split(regex);
			if(elems[0].equalsIgnoreCase(elems[1])) {
				continue;
			}
			List<Integer> inNode = new ArrayList<Integer>();
			if (!nodeMap.containsKey(Integer.parseInt(elems[0]))) {
				inNode.add(Integer.parseInt(elems[1]));
				nodeMap.put(Integer.parseInt(elems[0]), inNode);
			} else {
				inNode = nodeMap.get(Integer.parseInt(elems[0]));
				if(!inNode.contains(Integer.parseInt(elems[1]))) {
					inNode.add(Integer.parseInt(elems[1]));
					nodeMap.put(Integer.parseInt(elems[0]), inNode);
				}
			}
		}
		
		return nodeMap;
	}
	
	public static Map<Integer, Map<Integer, Double>> initEdgeProb(String filePath
								, String regex, Double initVal) throws IOException {
		
		Map<Integer, Map<Integer, Double>> edgeProb = new HashMap<Integer, Map<Integer,Double>>();
		
		BufferedReader br = FileUtil.getBufferReader(filePath);
		String line = null;
		while((line=br.readLine())!=null) {
			String[] elems = line.split(regex);
			if(elems[0].equalsIgnoreCase(elems[1])) {
				continue;
			}
			Map<Integer, Double> tmpMap;
			if (!edgeProb.containsKey(Integer.parseInt(elems[1]))) {
				tmpMap = new HashMap<Integer, Double>();
			} else {
				tmpMap = edgeProb.get(Integer.parseInt(elems[1]));
			}
			tmpMap.put(Integer.parseInt(elems[0]), initVal);
			edgeProb.put(Integer.parseInt(elems[1]), tmpMap);
		}
		
		return edgeProb;
	}
	
	public static Map<Integer, List<Integer>> getReverseNodeMap(String filePath
								, String regex) throws IOException {
		
		Map<Integer, List<Integer>> rvNodeMap = new HashMap<Integer, List<Integer>>();
		
		BufferedReader br = FileUtil.getBufferReader(filePath);
		String line = null;
		while((line=br.readLine())!=null) {
			String[] elems = line.split(regex);
			if(elems[0].equalsIgnoreCase(elems[1])) {
				continue;
			}
			List<Integer> outNode = new ArrayList<Integer>();
			if (!rvNodeMap.containsKey(Integer.parseInt(elems[1]))) {
				outNode.add(Integer.parseInt(elems[0]));
				rvNodeMap.put(Integer.parseInt(elems[1]), outNode);
			} else {
				outNode = rvNodeMap.get(Integer.parseInt(elems[1]));
				if(!outNode.contains(Integer.parseInt(elems[0]))) {
					outNode.add(Integer.parseInt(elems[0]));
					rvNodeMap.put(Integer.parseInt(elems[1]), outNode);
				}
			}
		}
		
		return rvNodeMap;
	}
	
}
