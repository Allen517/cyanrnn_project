/**   
 * @package	com.seocnnet.venus.core.util
 * @File		CollectionHelper.java
 * @Crtdate	May 26, 2012
 *
 * Copyright (c) 2012 by <a href="mailto:wangyongqing.casia@gmail.com">Allen Wang</a>.   
 */
package com.kingwang.netattrnn.comm.utils;

import java.util.Collection;

/**
 *
 * @author Allen Wang
 * 
 * May 26, 2012 11:35:01 AM
 * @version 1.0
 */
public class CollectionHelper {
	
	public static boolean isEmpty(Collection<?> set){
		if(null==set || set.isEmpty()){
			return true;
		}
		
		return false;
	}
}
