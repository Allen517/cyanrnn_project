README

#the architecture of "src" directory
-main.java.com.kingwang.netattrnn

--baselines
---evals
----RNNModelEvals.java: implementation of RNN validation in tranining process
---rnn
----RNN.java: main process of RNN

--batchderv: when minibatch is finished, batchderv will calculate the current derivation from all calculated derivation in each batch
---BatchDerivative.java: interface of BatchDerivative
---impl
----AttBatchDerivative.java: for attention layer
----AttWithCovBatchDerivative.java: for attention layer with coverage
----GRUBatchDerivative.java: for GRU (RNN)
----InputBatchDerivative.java: for input layer
----LSTMBatchDerivative.java: for LSTM (RNN)
----OutputBatchDerivative.java: for output layer
----OutputBatchWithHSoftmaxDerivative.java: for output layer with hierachical softmax
----OutputBatchWithOnlyTimeDerivative.java: for output layer (only calculating the generation of activated time)
----OutputBatchWithTimeDerivative.java: for output layer with hierachical softmax (calculating the generation of activated time and activated users)

--cells
---Cell.java: interface of RNN layers
---Operator.java: basic operator for RNN layers
---baselines
----rnn/impl: implementation of RNN
---impl: implementation of CYAN-RNN and CYAN-RNN(cov)
---main: main process of CYAN-RNN

--comm/utils: common utilities

--cons: constants

--dataset: implementation of loading dataset

--evals: implementation of CYAN-RNN and CYAN-RNN(cov) validation in tranining process

--utils: common utilities for RNN, CYAN-RNN, CYAN-RNN(cov)
