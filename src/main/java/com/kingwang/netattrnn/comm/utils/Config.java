/**   
 * @package	com.kingwang.rnncdm.commutils
 * @File		Config.java
 * @Crtdate	May 25, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.comm.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author King Wang
 * 
 * May 25, 2016 5:06:16 PM
 * @version 1.0
 */
public class Config {

	public static Map<String, String> getConfParams(String confPath) throws IOException {
		
		Map<String, String> confs = new HashMap<String, String>();
		
		BufferedReader br = FileUtil.getBufferReader(confPath);
		String line = null;
		while((line=br.readLine())!=null) {
			String[] elems = line.split("=");
			if(elems.length<2) {
				continue;
			}
			String key = elems[0].trim();
			String val = elems[1].trim();
			if(!confs.containsKey(key)) {
				confs.put(key, val);
			}
		}
		
		return confs;
	}
	
	public static void printConf(Map<String, String> conf, String logFile) {
		
		OutputStreamWriter oswLog = FileUtil.getOutputStreamWriter(logFile);
		
		for(Map.Entry<String, String> confEntry : conf.entrySet()) {
			System.out.println(confEntry.getKey()+":"+confEntry.getValue());
			FileUtil.writeln(oswLog, confEntry.getKey()+":"+confEntry.getValue());
		}
	}
}
