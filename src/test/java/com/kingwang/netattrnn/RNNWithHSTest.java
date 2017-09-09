/**   
 * @package	com.kingwang.rnncdm
 * @File		CellTest.java
 * @Crtdate	May 23, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.netattrnn;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Test;

import com.kingwang.netattrnn.cells.baselines.rnn.impl.GRURNN;
import com.kingwang.netattrnn.cells.baselines.rnn.impl.InputLayerRNN;
import com.kingwang.netattrnn.cells.baselines.rnn.impl.OutputLayerRNNWithHS;
import com.kingwang.netattrnn.cons.AlgConsHSoftmax;
import com.kingwang.netattrnn.utils.MatIniter;
import com.kingwang.netattrnn.utils.MatIniter.Type;

/**
 *
 * @author King Wang
 * 
 * May 23, 2016 10:16:59 AM
 * @version 1.0
 */
public class RNNWithHSTest {

	private int inDynSize;
	private int inFixedSize;
	private int outSize; 
	private int nodeSize;
	private int[] clsRcd;
	private InputLayerRNN input;
	private GRURNN gru;
	private OutputLayerRNNWithHS output;
	private Map<String, DoubleMatrix> nodeCode;
	private Map<String, DoubleMatrix> acts = new HashMap<>();
	private List<Double> y1_arr = new ArrayList<>();
	private List<Double> y0_arr = new ArrayList<>();
	
	@Before
	public void setCell() {
		// set basic parameters
		inDynSize = 10;
		inFixedSize = 1;
		outSize = 8;
		nodeSize = 5;
		clsRcd = new int[2];
		clsRcd[0] = 0;
		clsRcd[1] = 1;
		AlgConsHSoftmax.cNum = 2;
		AlgConsHSoftmax.nodeSizeInCls = new int[2];
		AlgConsHSoftmax.nodeSizeInCls[0] = 2;
		AlgConsHSoftmax.nodeSizeInCls[1] = 3;
		AlgConsHSoftmax.windowSize = 2;
		AlgConsHSoftmax.biasInitVal = 0;
		
		AlgConsHSoftmax.rnnType="gru";
		input = new InputLayerRNN(nodeSize, inDynSize, new MatIniter(Type.Test, 0, 0, 0));
		gru = new GRURNN(inDynSize, inFixedSize, outSize, new MatIniter(Type.Test, 0, 0, 0)); // set cell
		output = new OutputLayerRNNWithHS(outSize, AlgConsHSoftmax.cNum, new MatIniter(Type.Test, 0, 0, 0));
		
		DoubleMatrix Wx = new DoubleMatrix(nodeSize, inDynSize);
		nodeCode = new HashMap<>();
		DoubleMatrix code = new DoubleMatrix(1, nodeSize);
		nodeCode.put("0", code.put(0, 1));
		code = new DoubleMatrix(1, nodeSize);
		nodeCode.put("1", code.put(1, 1));
		code = new DoubleMatrix(1, nodeSize);
		nodeCode.put("2", code.put(2, 1));
	}
	
	/**
	 * Test method for {@link com.kingwang.ctsrnn.rnn.impl.LSTM.lstm.Cell#active(int, java.util.Map)}.
	 */
	@Test
	public void testActive() {
		// settings
		
		System.out.println("active function test");
//		DoubleMatrix code = nodeCode.get("1");
		DoubleMatrix code = new DoubleMatrix(1);
		code.put(0, 1);
		acts.put("code"+1, code);
		input.Wx.put(1, 2, 1);
		acts.put("fixedFeat"+1, DoubleMatrix.ones(1));
		
		DoubleMatrix prevS = DoubleMatrix.ones(1, outSize);
		acts.put("s"+0, prevS.mul(0.1));
		
		acts.put("r" + 0, DoubleMatrix.ones(1, 8).mul(0.1));
        acts.put("z" + 0, DoubleMatrix.ones(1, 8).mul(0.2));
        acts.put("gh" + 0, DoubleMatrix.ones(1, 8).mul(0.3));
        acts.put("h" + 0, DoubleMatrix.ones(1, 8).mul(0.1));
        
        input.active(1, acts);
        
		gru.active(1, acts);
		System.out.println(Math.pow(1+Math.exp(-.45), -1)+","+acts.get("r"+1).get(1, 2));
		assertEquals(Math.pow(1+Math.exp(-.45), -1), acts.get("r"+1).get(1, 2), 10e-3);
		System.out.println(Math.pow(1+Math.exp(-.45), -1)+","+acts.get("z"+1).get(1, 2));
		assertEquals(Math.pow(1+Math.exp(-.45), -1), acts.get("z"+1).get(1, 2), 10e-3);
		System.out.println(Math.tanh(.29+.16*Math.pow(1+Math.exp(-.45), -1))
							+","+acts.get("gh"+1).get(1, 2));
		assertEquals(Math.tanh(.29+.16*Math.pow(1+Math.exp(-.45), -1)), acts.get("gh"+1).get(1, 2), 10e-3);
		double z = Math.pow(1+Math.exp(-.45), -1);
		double gh = Math.tanh(.29+.16*Math.pow(1+Math.exp(-.45), -1));
		double h1 = z*.1+(1-z)*gh;
		System.out.println(h1);
		assertEquals(h1, acts.get("h"+1).get(1,2), 10e-3);
	}
	
	private void calcOneTurn(List<String> nodes) {
		
		for (int t=0; t<nodes.size()-1; t++) {
	    	String ndId = nodes.get(t);
	    	String nxtNdId = nodes.get(t+1);
	    	int nxtNdIdx = -1;
	
	    	DoubleMatrix fixedFeat = DoubleMatrix.zeros(1, inFixedSize);
			fixedFeat.put(0, 1);
	    	
//	    	DoubleMatrix code = nodeCode.get(ndId);
			DoubleMatrix code = new DoubleMatrix(1);
			code.put(0, Double.parseDouble(ndId));
	    	acts.put("code"+t, code);
	    	acts.put("fixedFeat"+t, fixedFeat);
	    	
	    	int nodeCls = clsRcd[t];
	    	
	    	input.active(t, acts);
	        gru.active(t, acts);
	        output.active(t, acts, nodeCls);
	       
//          DoubleMatrix hatYt = output.yDecode(acts.get("s" + t));
//	        DoubleMatrix predictYt = Activer.softmax(hatYt);
//	        acts.put("py" + t, predictYt);
	        DoubleMatrix y = new DoubleMatrix(1, AlgConsHSoftmax.nodeSizeInCls[nodeCls]);
	        if(nxtNdId.equalsIgnoreCase("1")) {
	        	nxtNdIdx = 0;
	        }
	        if(nxtNdId.equalsIgnoreCase("0")) {
	        	nxtNdIdx = 1;
	        }
	        y.put(nxtNdIdx, 1);
	        DoubleMatrix cls = new DoubleMatrix(1, AlgConsHSoftmax.cNum);
	        cls.put(nodeCls, 1);
	        acts.put("y" + t, y);
	        acts.put("cls"+t, cls);
		}
	}
	
	private void gradientTestAndretActualGradient(List<String> nodes, DoubleMatrix mat
													, int reviseLoc, int targetT, double delta) {
		
		y0_arr = new ArrayList<>();
		y1_arr = new ArrayList<>();
		
		acts.clear();
		mat = mat.put(reviseLoc, mat.get(reviseLoc)-delta); // reset Wxi
		calcOneTurn(nodes);
		for(int t=0; t<targetT+1; t++) {
			String nxtNdId = nodes.get(t+1);
	    	int nxtNdIdx = -1;
	    	int nodeCls = clsRcd[t];
	    	if(nxtNdId.equalsIgnoreCase("1")) {
	        	nxtNdIdx = 0;
	        }
	        if(nxtNdId.equalsIgnoreCase("0")) {
	        	nxtNdIdx = 1;
	        }
			DoubleMatrix py = acts.get("py" + t);
			DoubleMatrix pc = acts.get("pc" + t);
			y1_arr.add(Math.log(py.get(nxtNdIdx))+Math.log(pc.get(nodeCls)));
		}
		//original
		acts.clear();
		mat = mat.put(reviseLoc, mat.get(reviseLoc)+2*delta);
		calcOneTurn(nodes);
		for(int t=0; t<targetT+1; t++) {
			String nxtNdId = nodes.get(t+1);
	    	int nxtNdIdx = -1;
	    	int nodeCls = clsRcd[t];
	    	if(nxtNdId.equalsIgnoreCase("1")) {
	        	nxtNdIdx = 0;
	        }
	        if(nxtNdId.equalsIgnoreCase("0")) {
	        	nxtNdIdx = 1;
	        }
			DoubleMatrix py = acts.get("py" + t);
			DoubleMatrix pc = acts.get("pc" + t);
			y0_arr.add(Math.log(py.get(nxtNdIdx))+Math.log(pc.get(nodeCls)));
		}
		 
		// test
		acts.clear();
		mat = mat.put(reviseLoc, mat.get(reviseLoc)-delta); // set back to the original Wxi
		calcOneTurn(nodes);
		output.bptt(acts, targetT);
		gru.bptt(acts, targetT, output);
		input.bptt(acts, targetT, gru);
	}
	
	/**
	 * Test method for {@link com.kingwang.ctsrnn.rnn.impl.LSTM.lstm.Cell#bptt(java.util.List, java.util.Map, java.util.Map, int, double)}.
	 */
	@Test
	public void testBptt() {
		
		inDynSize = 3;
		inFixedSize = 1;
		outSize = 2;
		nodeSize = 5;
		
		double delta = 10e-7;
		
		// set input
		List<String> ndList = new ArrayList<>();
		ndList.add("2");
		ndList.add("1");
		ndList.add("0");
		
//		AlgConsHSoftmax.nodeSizeInCls = new int[2];
//		AlgConsHSoftmax.nodeSizeInCls[0] = 0;
//		AlgConsHSoftmax.nodeSizeInCls[0] = 1;
		
//		List<Double> tmList = new ArrayList<>();
//		tmList.add(2.);
//		tmList.add(3.);
		
		DoubleMatrix fixedFeat = DoubleMatrix.zeros(1, inFixedSize);
		fixedFeat.put(0, 1);
//		input.tmFeat.put(1, .5);
//		DoubleMatrix x = new DoubleMatrix(1, 10);
//		x.put(2, 1);
//		nodeVec.put(3, x);
//		nodeVec.put(2, x);
//		nodeVec.put(1, x);
		
		MatIniter initer = new MatIniter(Type.Uniform, 1, 0, 0);
		
		input = new InputLayerRNN(nodeSize, inDynSize, initer);
		gru = new GRURNN(inDynSize, inFixedSize, outSize, initer);
		output = new OutputLayerRNNWithHS(outSize, AlgConsHSoftmax.cNum, initer);
		int reviseLoc = 1;
		int targetT = 1;
		
		/**
		 * Wxr
		 */
		System.out.println("Wxr test");
		gradientTestAndretActualGradient(ndList, gru.Wxr, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxr_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxr_2 += tmp/2/delta;
		}
		System.out.println(deltaWxr_2+","+(-acts.get("dWxr").get(reviseLoc)));
		assertEquals(deltaWxr_2, -acts.get("dWxr").get(reviseLoc), 10e-7);
		
		/**
		 * Wdr
		 */
		reviseLoc = 0;
		System.out.println("Wdr test");
		gradientTestAndretActualGradient(ndList, gru.Wdr, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWdr_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWdr_2 += tmp/2/delta;
		}
		System.out.println(deltaWdr_2+","+(-acts.get("dWdr").get(reviseLoc)));
		assertEquals(deltaWdr_2, -acts.get("dWdr").get(reviseLoc), 10e-7);
		
		/**
		 * Whr
		 */
		reviseLoc = 0;
		System.out.println("Whr test");
		gradientTestAndretActualGradient(ndList, gru.Whr, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWhr_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWhr_2 += tmp/2/delta;
		}
		System.out.println(deltaWhr_2+","+(-acts.get("dWhr").get(reviseLoc)));
		assertEquals(deltaWhr_2, -acts.get("dWhr").get(reviseLoc), 10e-7);
		
		/**
		 * br
		 */
		reviseLoc = 0;
		System.out.println("br test");
		gradientTestAndretActualGradient(ndList, gru.br, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltabr_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltabr_2 += tmp/2/delta;
		}
		System.out.println(deltabr_2+","+(-acts.get("dbr").get(reviseLoc)));
		assertEquals(deltabr_2, -acts.get("dbr").get(reviseLoc), 10e-7);
		
		/**
		 * Wxh
		 */
		System.out.println("Wxh test");
		gradientTestAndretActualGradient(ndList, gru.Wxh, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxh_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxh_2 += tmp/2/delta;
		}
		System.out.println(deltaWxh_2+","+(-acts.get("dWxh").get(reviseLoc)));
		assertEquals(deltaWxh_2, -acts.get("dWxh").get(reviseLoc), 10e-7);
		
		/**
		 * Wdh
		 */
		System.out.println("Wdh test");
		gradientTestAndretActualGradient(ndList, gru.Wdh, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWdh_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWdh_2 += tmp/2/delta;
		}
		System.out.println(deltaWdh_2+","+(-acts.get("dWdh").get(reviseLoc)));
		assertEquals(deltaWdh_2, -acts.get("dWdh").get(reviseLoc), 10e-7);
		
		/***
		 * Whh
		 */
		System.out.println("Whh test");
		gradientTestAndretActualGradient(ndList, gru.Whh, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWhh_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWhh_2 += tmp/2/delta;
		}
		System.out.println(deltaWhh_2+","+(-acts.get("dWhh").get(reviseLoc)));
		assertEquals(deltaWhh_2, -acts.get("dWhh").get(reviseLoc), 10e-7);
		
		/**
		 * bh
		 */
		System.out.println("bh test");
		gradientTestAndretActualGradient(ndList, gru.bh, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltabh_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltabh_2 += tmp/2/delta;
		}
		System.out.println(deltabh_2+","+(-acts.get("dbh").get(reviseLoc)));
		assertEquals(deltabh_2, -acts.get("dbh").get(reviseLoc), 10e-7);
		
		/***
		 * Wxz
		 */
		System.out.println("Wxz test");
		gradientTestAndretActualGradient(ndList, gru.Wxz, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxz_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxz_2 += tmp/2/delta;
		}
		System.out.println(deltaWxz_2+","+(-acts.get("dWxz").get(reviseLoc)));
		assertEquals(deltaWxz_2, -acts.get("dWxz").get(reviseLoc), 10e-7);
		
		/***
		 * Wdz
		 */
		System.out.println("Wdz test");
		gradientTestAndretActualGradient(ndList, gru.Wdz, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWdz_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWdz_2 += tmp/2/delta;
		}
		System.out.println(deltaWdz_2+","+(-acts.get("dWdz").get(reviseLoc)));
		assertEquals(deltaWdz_2, -acts.get("dWdz").get(reviseLoc), 10e-7);
		
		/***
		 * Whz
		 */
		System.out.println("Whz test");
		gradientTestAndretActualGradient(ndList, gru.Whz, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWhz_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWhz_2 += tmp/2/delta;
		}
		System.out.println(deltaWhz_2+","+(-acts.get("dWhz").get(reviseLoc)));
		assertEquals(deltaWhz_2, -acts.get("dWhz").get(reviseLoc), 10e-7);
		
		/**
		 * bz
		 */
		System.out.println("bz test");
		gradientTestAndretActualGradient(ndList, gru.bz, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltabz_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltabz_2 += tmp/2/delta;
		}
		System.out.println(deltabz_2+","+(-acts.get("dbz").get(0)));
		assertEquals(deltabz_2, -acts.get("dbz").get(0), 10e-7);
		
//		/**
//		 * Whd
//		 */
//		System.out.println("Whd test");
//		gradientTestAndretActualGradient(ndList, tmList, cell.Whd, reviseLoc, targetT, delta);
//		// get the actual partial y/partial x
//		double deltaWhd_2 = 0;
//		for(int t=0; t<targetT+1; t++) {
//			double tmp = y0_arr.get(t)-y1_arr.get(t);
//			deltaWhd_2 += tmp/2/delta;
//		}
//		System.out.println(deltaWhd_2+","+(-acts.get("dWhd").get(reviseLoc)));
//		assertEquals(deltaWhd_2, -acts.get("dWhd").get(reviseLoc), 10e-7);
//		
//		/**
//		 * bd
//		 */
//		System.out.println("bd test");
//		gradientTestAndretActualGradient(ndList, tmList, cell.bd, reviseLoc, targetT, delta);
//		// get the actual partial y/partial x
//		double deltabd_2 = 0;
//		for(int t=0; t<targetT+1; t++) {
//			double tmp = y0_arr.get(t)-y1_arr.get(t);
//			deltabd_2 += tmp/2/delta;
//		}
//		System.out.println(deltabd_2+","+(-acts.get("dbd").get(0)));
//		assertEquals(deltabd_2, -acts.get("dbd").get(0), 10e-7);
		
		/**
		 * Wsy
		 */
		System.out.println("Wsy test");
		gradientTestAndretActualGradient(ndList, output.Wsy[0], reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWhy_2 = 0;
		double[] tmpy = new double[targetT+1];
		for(int t=0; t<targetT+1; t++) {
	    	int nodeCls = clsRcd[t];
			tmpy[nodeCls] += y0_arr.get(t)-y1_arr.get(t);
		}
		deltaWhy_2 = tmpy[0]/2/delta;
		System.out.println(deltaWhy_2+","+(-acts.get("dWsy0").get(reviseLoc)));
		assertEquals(deltaWhy_2, -acts.get("dWsy0").get(reviseLoc), 10e-7);
		
		/**
		 * by
		 */
		System.out.println("by test");
		gradientTestAndretActualGradient(ndList, output.by[1], reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaby_2 = 0;
		tmpy = new double[targetT+1];
		for(int t=0; t<targetT+1; t++) {
			int nodeCls = clsRcd[t];
			tmpy[nodeCls] += y0_arr.get(t)-y1_arr.get(t);
		}
		deltaby_2 += tmpy[1]/2/delta;
		System.out.println(deltaby_2+","+(-acts.get("dby1").get(reviseLoc)));
		assertEquals(deltaby_2, -acts.get("dby1").get(reviseLoc), 10e-7);
		
		/**
		 * Wx
		 */
		System.out.println("Wx test");
		gradientTestAndretActualGradient(ndList, input.Wx, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWx_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWx_2 += tmp/2/delta;
		}
		System.out.println(deltaWx_2+","+(-acts.get("dWx").get(reviseLoc)));
		assertEquals(deltaWx_2, -acts.get("dWx").get(reviseLoc), 10e-7);
		
		/**
		 * bx
		 */
		System.out.println("bx test");
		gradientTestAndretActualGradient(ndList, input.bx, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltabx_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltabx_2 += tmp/2/delta;
		}
		System.out.println(deltabx_2+","+(-acts.get("dbx").get(reviseLoc)));
		assertEquals(deltabx_2, -acts.get("dbx").get(reviseLoc), 10e-7);
	}
	
//
//	/**
//	 * Test method for {@link com.kingwang.rnncdm.lstm.Cell#decode(org.jblas.DoubleMatrix)}.
//	 */
//	@Test
//	public void testDecode() {
//		fail("Not yet implemented");
//	}
//
//	/**
//	 * Test method for {@link com.kingwang.rnncdm.lstm.Cell#loadRNNModel(java.lang.String)}.
//	 */
//	@Test
//	public void testLoadRNNModel() {
//		fail("Not yet implemented");
//	}

}
