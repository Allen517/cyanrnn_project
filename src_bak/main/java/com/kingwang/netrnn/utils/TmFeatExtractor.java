/**   
 * @package	com.kingwang.cdmrnn.rnn
 * @File		TmFeatExtractor.java
 * @Crtdate	Aug 18, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netrnn.utils;

import java.util.Calendar;
import java.util.Date;

import org.jblas.DoubleMatrix;

import com.kingwang.netrnn.cons.AlgConsHSoftmax;

/**
 *
 * @author King Wang
 * 
 * Aug 18, 2016 11:16:51 PM
 * @version 1.0
 */
public class TmFeatExtractor {

	public static DoubleMatrix timeFeatExtractor(double curTm, double prevTm) {
		
		DoubleMatrix tmFeat = new DoubleMatrix(1, 136);
		
		if(curTm>0) {
			Calendar cal = Calendar.getInstance();
			Date curDate = new Date((long) (curTm*3600*1000));
			cal.setTime(curDate);
//			//year 2008~2016=>9 years
//			System.out.println(curDate);
//			int yearIdx = cal.get(Calendar.YEAR)%2000-8;
//			System.out.println(cal.get(Calendar.YEAR));
//			tmFeat.put(yearIdx, 1);
			//month 1~12=>12 months
			int monthIdx = cal.get(Calendar.MONTH);
			tmFeat.put(monthIdx, 1); //year
			//day 1~31=>31 days
			int dayIdx = cal.get(Calendar.DAY_OF_MONTH)%31;
			tmFeat.put(12+dayIdx, 1);
			//hours 1-24=>24 hours
			int hrIdx = cal.get(Calendar.HOUR_OF_DAY)%24;
			tmFeat.put(43+hrIdx, 1);
			//minutes 1~60=>60
			int minuteIdx = cal.get(Calendar.MINUTE)%60;
			tmFeat.put(67+minuteIdx, 1);
			//weekdays 1~7=>7 days
			int weekdayIdx = cal.get(Calendar.DAY_OF_WEEK)%7;
			tmFeat.put(127+weekdayIdx, 1);
		}
		
		if(prevTm>0) {
			double tmInterval = (curTm-prevTm)/AlgConsHSoftmax.tmDiv;
			if(tmInterval>0) {
				tmFeat.put(134, Math.log(tmInterval));
				tmFeat.put(135, tmInterval);
			}
		} else {
			tmFeat.put(134, 0);
			tmFeat.put(135, 0);
		}
		
		return tmFeat;
	}
}
