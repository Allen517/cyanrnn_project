/**   
 * @package	com.kingwang.rnncdm.utils
 * @File		WeightTypes.java
 * @Crtdate	May 22, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn.utils;

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
	Wxt("Wxt"), Wdt("Wdt"), Wtt("Wtt"), Wst("Wst"), bt("bt"), 
	V("V"), U("U"), W("W"), Z("Z"), bs("bs"),
	//for coverage
	Wvv("Wvv"), Wav("Wav"), Whv("Whv"), Wtv("Wtv"), bv("bv"),
	//for output
	Wxy("Wxy"), Wdy("Wdy"), Wty("Wty"), Wsy("Wsy"), by("by"), 
	Wxc("Wxc"), Wdc("Wdc"), Wtc("Wtc"),Wsc("Wsc"), bsc("bsc"),
	Wxd("Wxd"), Wdd("Wdd"), Wsd("Wsd"), Wtd("Wtd"), bd("bd"), Wd("Wd"),
	//for input
	Wx("Wx"), bx("bx"),
	Null("Null");
	
	private final String strVal; 
	
	private LoadTypes(final String strVal) {
		this.strVal = strVal;
	}
}
