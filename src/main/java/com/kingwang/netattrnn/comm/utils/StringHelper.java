/**
 * 
 */
package com.kingwang.netattrnn.comm.utils;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class StringHelper {
	/**
	 * 字符串中的单双引号转换和换行符替换
	 * 
	 * @param str
	 * @return
	 */
	public static String transForJS(String str) {
		String str1 = str.replaceAll("\"", "'");
		String str2 = str1.replaceAll("\\n", "<br>");
		return str2;
	}	
	
	/**
	 * 将字符串数组转化为整型数组
	 * 
	 * @param str
	 * @return
	 */
	public static Long[] castToLong(String[] str) {
		if (str == null || str.length == 0) {
			return new Long[0];
		}
		Long[] is = new Long[str.length];
		for (int i = 0; i < str.length; i++) {
			is[i] = Long.valueOf(str[i]);
		}
		return is;
	}

	/**
	 * 将常整型数组转换成字符串数组
	 * 
	 * @param ls
	 * @return
	 */
	public static String[] castToString(Long[] ls) {
		if (ls == null || ls.length == 0) {
			return new String[0];
		}
		String[] str = new String[ls.length];
		for (int i = 0; i < ls.length; i++) {
			str[i] = ls[i].toString();
		}
		return str;
	}

	/**
	 * 从一个List中抽取出不同的元素组成一个List<Map>
	 * 
	 * @param list
	 * @param value
	 * @return
	 */
	public static List<Map<Long, Long>> extractMapFromList(List<Long> list, long value) {
		List<Map<Long, Long>> result = new ArrayList<Map<Long, Long>>();
		Map<Long, Long> taMap = new HashMap<Long, Long>();
		for (int i = 0; i < list.size(); i++) {
			long long1 = list.get(i);
			if (!taMap.containsKey(long1)) {
				taMap.put(long1, value);
				if (i == (list.size() - 1)) {
					result.add(taMap);
				}
			} else {
				result.add(taMap);
				taMap = new HashMap<Long, Long>();
				taMap.put(long1, value);
				if (i == (list.size() - 1)) {
					result.add(taMap);
				}
			}
		}
		return result;
	}
	
	public static boolean isValidTelephone(String s) {
		return s != null && s.matches("^[0-9]{6}[0-9]*$");
	}
	
	public static boolean isValidCellphone(String s) {
		return s != null && s.matches("^[0-9]{11}$");
	}
	
	public static boolean isValidIdCardNumber(String s) {
		return s != null && s.matches("^([0-9]{15})|([0-9]{17}[0-9xX])$");
	}
	
	public static boolean isValidMailBox(String s) {
		return s != null && s.matches("^[\\w]+(.[\\w-]+)*@[\\w-]+(.[\\w-]+)+$");
	}
	
	public static boolean isValidDate(String s){
		return s != null && s.matches("^\\d{4}(\\-)\\d{1,2}((\\-)\\d{1,2})?$");
	}
	
	public static String transCharForMysql(String query) {
		String result = query.replaceAll("-", "\\\\-");
		result = result.replaceAll("_", "\\\\_");
		result = result.replaceAll("%", "\\\\%");
		result = result.replaceAll("'", "\\\\'");
		result = result.replaceAll("\"", "\\\\\"");
		return result;
	}
	
	/**
	 * 
	 * description:
	 * 判断string对象是否为空，可能的情况有："", "  ", null
	 *
	 * @param s
	 * @return
	 */
	public static boolean isEmpty(String s){
		if(null == s || s.trim().equals("")){
			return true;
		}
		return false;
	}
	
	public static String getDirPathWithoutSep(String filePath) {
		
		if(filePath.substring(filePath.length()-1, filePath.length()).equalsIgnoreCase(File.separator)) {
			return filePath.substring(0, filePath.length()-1);
		}
		
		return filePath;
	}
	
}
