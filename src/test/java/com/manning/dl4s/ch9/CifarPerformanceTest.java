package com.manning.dl4s.ch9;

import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Tests for evaluations of different CNN architectures over Cifar dataset
 */
public class CifarPerformanceTest {

  @Test
  public void testCifar24() {

    int height = 32;
    int width = 32;
    int channels = 3;
    int numLabels = CifarLoader.NUM_LABELS;
    int numSamples = 5000;
    int batchSize = 24;
    int freIterations = 50;
    int seed = 123;
    boolean preProcessCifar = false;
    int epochs = 1;

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .list()
        .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}).convolutionMode(ConvolutionMode.Same)
            .nIn(3).nOut(20).activation(Activation.RELU).build())
        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).build())
        .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {0,0}).convolutionMode(ConvolutionMode.Same)
            .nOut(10).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
            .biasInit(1e-2).build())
        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).build())
        .layer(4, new DenseLayer.Builder().nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(numLabels).activation(Activation.SOFTMAX).build())
        .backprop(true)
        .pretrain(false)
        .setInputType(InputType.convolutional(height, width, channels))
        .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    model.setListeners(new ScoreIterationListener(freIterations));

    CifarDataSetIterator dsi = new CifarDataSetIterator(batchSize, numSamples, new int[] {height, width, channels}, preProcessCifar, true);
    CifarDataSetIterator cifarEval = new CifarDataSetIterator(batchSize, 1000,
        new int[] {height, width, channels}, preProcessCifar, false);

    System.out.println(String.join(",", dsi.getLabels().toArray(new String[dsi.getLabels().size()])));

    for (int i = 0; i < epochs; ++i) {
      model.fit(dsi);
    }

    Evaluation eval = new Evaluation(cifarEval.getLabels());
    while(cifarEval.hasNext()) {
      DataSet testDS = cifarEval.next(batchSize);
      INDArray output = model.output(testDS.getFeatures());
      eval.eval(testDS.getLabels(), output);
    }
    System.out.println(eval.stats());

  }

  @Test
  public void testCifarBest() throws Exception {

    int height = 32;
    int width = 32;
    int channels = 3;
    int numLabels = CifarLoader.NUM_LABELS;
    int numSamples = 5000;
    int batchSize = 24;
    int freIterations = 50;
    int seed = 123;
    boolean preProcessCifar = false;
    int epochs = 1;

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .cacheMode(CacheMode.DEVICE).updater(new Adam(0.001D)).biasUpdater(new Adam(0.02D)).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l1(1.0E-4D).l2(5.0E-4D)
        .list()
        .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}).convolutionMode(ConvolutionMode.Same)
            .nIn(3).nOut(20).activation(Activation.RELU).build())
        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).build())
        .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {0,0}).convolutionMode(ConvolutionMode.Same)
            .nOut(10).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
            .biasInit(1e-2).build())
        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).build())
        .layer(4, new DenseLayer.Builder().nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(numLabels).activation(Activation.SOFTMAX).build())
        .backprop(true)
        .pretrain(false)
        .setInputType(InputType.convolutional(height, width, channels))
        .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    model.setListeners(new ScoreIterationListener(freIterations));

    CifarDataSetIterator dsi = new CifarDataSetIterator(batchSize, numSamples, new int[] {height, width, channels}, preProcessCifar, true);
    CifarDataSetIterator cifarEval = new CifarDataSetIterator(batchSize, 1000,
        new int[] {height, width, channels}, preProcessCifar, false);

    System.out.println(String.join(",", dsi.getLabels().toArray(new String[dsi.getLabels().size()])));

    for (int i = 0; i < epochs; ++i) {
      model.fit(dsi);
    }

    Evaluation eval = new Evaluation(cifarEval.getLabels());
    while(cifarEval.hasNext()) {
      DataSet testDS = cifarEval.next(batchSize);
      INDArray output = model.output(testDS.getFeatures());
      eval.eval(testDS.getLabels(), output);
    }
    System.out.println(eval.stats());

  }

  @Test
  public void testCifarBestFull() throws Exception {

    int height = 32;
    int width = 32;
    int channels = 3;
    int numLabels = CifarLoader.NUM_LABELS;
    int numSamples = 50000;
    int batchSize = 24;
    int freIterations = 50;
    int seed = 123;
    boolean preProcessCifar = false;
    int epochs = 1;


    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .cacheMode(CacheMode.DEVICE).updater(new Adam(0.001D)).biasUpdater(new Adam(0.02D)).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l1(1.0E-4D).l2(5.0E-4D)
        .list()
        .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}).convolutionMode(ConvolutionMode.Same)
            .nIn(3).nOut(20).activation(Activation.RELU).build())
        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).build())
        .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {0,0}).convolutionMode(ConvolutionMode.Same)
            .nOut(10).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
            .biasInit(1e-2).build())
        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).build())
        .layer(4, new DenseLayer.Builder().nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(numLabels).activation(Activation.SOFTMAX).build())
        .backprop(true)
        .pretrain(false)
        .setInputType(InputType.convolutional(height, width, channels))
        .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    model.setListeners(new ScoreIterationListener(freIterations));

    CifarDataSetIterator dsi = new CifarDataSetIterator(batchSize, numSamples, new int[] {height, width, channels}, preProcessCifar, true);
    CifarDataSetIterator cifarEval = new CifarDataSetIterator(batchSize, 10000,
        new int[] {height, width, channels}, preProcessCifar, false);

    System.out.println(String.join(",", dsi.getLabels().toArray(new String[dsi.getLabels().size()])));

    for (int i = 0; i < epochs; ++i) {
      model.fit(dsi);
    }

    Evaluation eval = new Evaluation(cifarEval.getLabels());
    while(cifarEval.hasNext()) {
      DataSet testDS = cifarEval.next(batchSize);
      INDArray output = model.output(testDS.getFeatures());
      eval.eval(testDS.getLabels(), output);
    }
    System.out.println(eval.stats());

  }

  @Test
  public void testCifarFull24() throws Exception {

    int height = 32;
    int width = 32;
    int channels = 3;
    int numLabels = CifarLoader.NUM_LABELS;
    int numSamples = 50000;
    int batchSize = 24;
    int freIterations = 50;
    int seed = 123;
    boolean preProcessCifar = false;
    int epochs = 1;


    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .list()
        .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}).convolutionMode(ConvolutionMode.Same)
            .nIn(3).nOut(20).activation(Activation.RELU).build())
        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).stride(2,2).build())
        .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {0,0}).convolutionMode(ConvolutionMode.Same)
            .nOut(10).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
            .biasInit(1e-2).build())
        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).build())
        .layer(4, new DenseLayer.Builder().nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(numLabels).activation(Activation.SOFTMAX).build())
        .backprop(true)
        .pretrain(false)
        .setInputType(InputType.convolutional(height, width, channels))
        .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    model.setListeners(new ScoreIterationListener(freIterations));

    CifarDataSetIterator dsi = new CifarDataSetIterator(batchSize, numSamples, new int[] {height, width, channels}, preProcessCifar, true);
    CifarDataSetIterator cifarEval = new CifarDataSetIterator(batchSize, 10000,
        new int[] {height, width, channels}, preProcessCifar, false);

    System.out.println(String.join(",", dsi.getLabels().toArray(new String[dsi.getLabels().size()])));

    for (int i = 0; i < epochs; ++i) {
      model.fit(dsi);
    }

    Evaluation eval = new Evaluation(cifarEval.getLabels());
    while(cifarEval.hasNext()) {
      DataSet testDS = cifarEval.next(batchSize);
      INDArray output = model.output(testDS.getFeatures());
      eval.eval(testDS.getLabels(), output);
    }
    System.out.println(eval.stats());

  }
}
