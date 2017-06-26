/**   
 * @package	com.kingwang.rnncdm.utils
 * @File		WeightTypes.java
 * @Crtdate	May 22, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn.utils;

/**
 *
 * @author King Wang
 * 
 * May 22, 2016 5:07:48 PM
 * @version 1.0
 */
public enum LoadTypes {

	//for gru
	Wxr("Wxr"), Wdr("Wdr"), Whr("Whr"), br("br"), 
	Wxz("Wxz"), Wdz("Wdz"), Whz("Whz"), bz("bz"), 
	Wxh("Wxh"), Wdh("Wdh"), Whh("Whh"), bh("bh"),
	//for attention
	V("V"), U("U"), W("W"), bs("bs"),
	//for output
	Wxy("Wxy"), Wdy("Wdy"),
	Wsy("Wsy"), by("by"), 
	Wxc("Wxc"), Wdc("Wdc"),
	Wsc("Wsc"), bsc("bsc"),
	//for input
	Wx("Wx"), bx("bx"),
	Null("Null");
	
	private final String strVal; 
	
	private LoadTypes(final String strVal) {
		this.strVal = strVal;
	}
}
