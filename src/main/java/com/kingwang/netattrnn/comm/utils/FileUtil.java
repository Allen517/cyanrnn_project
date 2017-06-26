/**   
 * @package	anaylsis.data.util
 * @File		FileReader.java
 * @Crtdate	Jun 17, 2012
 *
 * Copyright (c) 2012 by <a href="mailto:wangyongqing.casia@gmail.com">Allen Wang</a>.   
 */
package com.kingwang.netattrnn.comm.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 
 * @author Allen Wang
 * 
 *         Jun 17, 2012 6:18:22 PM
 * @version 1.0
 */
public class FileUtil {

	public static BufferedReader getBufferReader(String filePath) {
		InputStreamReader isr = null;
		try {
			isr = new InputStreamReader(new FileInputStream(filePath),"utf8");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		if (isr == null) {
			return null;
		}

		return new BufferedReader(isr);
	}
	
	public static OutputStreamWriter getOutputStreamWriter(String filePath) {
		
		return getOutputStreamWriter(filePath, false);
	}

	public static OutputStreamWriter getOutputStreamWriter(String filePath, Boolean isAppend) {

		OutputStreamWriter osw = null;
		try {
			osw = new OutputStreamWriter(new FileOutputStream(filePath, isAppend), "utf8");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return osw;
	}
	
	public static boolean isCompleteLine(String line){
		String lineReg = "^\\d+\\t\\d+";
		
		Pattern p = Pattern.compile(lineReg);

		Matcher m = p.matcher(line);
		if (m.find()) {
			return true;
		} 
		
		return false;
	}
	
	public static void cleanLineStr(String line){
		String[] filterList = { "\"" };
		for (String str : filterList) {
			line = line.replaceAll(str, "");
		}
	}

	public static void fileFormatClean(String rdFilePath, String opFilePath) {
		OutputStreamWriter osw = getOutputStreamWriter(opFilePath, false);

		BufferedReader br = getBufferReader(rdFilePath);
		String line = null;
		String newLinePre = null;
		String newLine = null;
		boolean isNewLine = false;
		boolean isFirst = true;


		try {
			while ((line = br.readLine()) != null) {
				if (isCompleteLine(line)) {
					newLinePre = newLine;
					newLine = line;
					isNewLine = true;
				} else {
					newLine += line;
					isNewLine = false;
					isFirst = false;
				}
				if (isNewLine && !isFirst) {
					cleanLineStr(newLinePre);
					osw.append(newLinePre);
					osw.append(System.getProperty("line.separator"));
					osw.flush();
				}
			}
			cleanLineStr(newLine);
			osw.append(newLine);
			osw.append(System.getProperty("line.separator"));
			osw.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static String findLine(String line, String str) {
		String lowerCaseLine = line.toLowerCase()+"\t";
		String lowerCaseStr = str.toLowerCase();

		if (lowerCaseLine.contains(lowerCaseStr)) {
			return line;
		}

		return null;
	}

	public static String findLineExact(String line, String str) {
		if (line.contains("\t" + str)) {
			return line;
		}

		return null;
	}

	public static String getDesFilePath(String desDir, String filename, String type) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyMMddHHmmss");

		return desDir + File.separatorChar + filename +
				 "_" + sdf.format(Calendar.getInstance().getTime()) + type;
	}
	
	public static String getFileString(String path){
		BufferedReader br = getBufferReader(path);
		
		String fileStr = new String();
		String line = null;
		try {
			while((line=br.readLine())!=null){
				fileStr+=line+System.getProperty("line.separator");
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return fileStr;
	}
	
	public static void writeln(OutputStreamWriter osw, String str){
		try {
			osw.append(str);
			osw.append(System.getProperty("line.separator"));
			osw.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public synchronized static void exceptionOutput(String info, String type) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyMMdd");
		
		OutputStreamWriter osw = getOutputStreamWriter("./"+type+"_log_"
								+sdf.format(Calendar.getInstance().getTime())+".txt", true);
		writeln(osw, info);
		try {
			osw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public synchronized static void runtimeOutput(String info, String type) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyMMdd");
		
		OutputStreamWriter osw = getOutputStreamWriter("./"+type+"_runtime_"
				+sdf.format(Calendar.getInstance().getTime())+".txt", true);
		writeln(osw, info);
		try {
			osw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static List<String> getFileList(String fileDir) {
		
		List<String> filePathList = new ArrayList<String>();
		File f = new File(fileDir);
		File[] fileList = f.listFiles();
		List<File> dirList = new ArrayList<File>();
		for(File file : fileList) {
			if(file.isFile()) {
				filePathList.add(file.getPath());
			} else if(file.isDirectory()) {
				dirList.add(file);
			}
		}
		
		while(!CollectionHelper.isEmpty(dirList)) {
			
			File otherFile = new File(dirList.remove(0).getPath());
			File[] otherFileList = otherFile.listFiles();
			for(File file : otherFileList) {
				if(file.isFile()) {
					filePathList.add(file.getPath());
				} else if(file.isDirectory()) {
					dirList.add(file);
				}
			}
		}
		
		return filePathList;
	}
	
	public static Map<String, String> getConfParams(String confPath) throws IOException {
		
		Map<String, String> confs = new HashMap<String, String>();
		
		BufferedReader br = FileUtil.getBufferReader(confPath);
		String line = null;
		while((line=br.readLine())!=null) {
			String[] elems = line.split("=");
			if(elems.length<2) {
				continue;
			}
			String key = elems[0];
			String val = elems[1];
			if(!confs.containsKey(key)) {
				confs.put(key, val);
			}
		}
		
		return confs;
	}
	
	public static String getFileName(String filePath) {
		
		if(StringHelper.isEmpty(filePath)) {
			return null;
		}
		
		int fileSepIdx = filePath.lastIndexOf(File.separatorChar);
		
		return filePath.substring(fileSepIdx+1);
	}
}
