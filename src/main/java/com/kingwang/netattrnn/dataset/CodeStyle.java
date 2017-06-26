/**   
 * @package	com.kingwang.cdmrnn.dataset
 * @File		CodeStyle.java
 * @Crtdate	Oct 24, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.dataset;

/**
 *
 * @author King Wang
 * 
 * Oct 24, 2016 11:35:48 AM
 * @version 1.0
 */
public enum CodeStyle {

	SINGLE("single"), BIN("bin"), ONEHOT("onehot");
	
	private String codeName;
	
	private CodeStyle(String codeName) {
		this.codeName = codeName;
	}
}
