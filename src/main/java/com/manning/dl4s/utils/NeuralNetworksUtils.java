package com.manning.dl4s.utils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Varius utility methods for NNs
 */
public class NeuralNetworksUtils {

  private static final Logger log = LoggerFactory.getLogger(NeuralNetworksUtils.class);

  private static final Random rng = new Random();

  private static double d;

  public static Map<String, Double> sampleFromNetwork(MultiLayerNetwork network, CharacterIterator characterIterator, String initialization, int numSamples, Character eosChar) {

    if (initialization == null) {
      initialization = String.valueOf(characterIterator.convertIndexToCharacter(rng.nextInt(characterIterator.inputColumns())));
    }

    StringBuilder[] sb = new StringBuilder[numSamples];
    for (int i = 0; i < numSamples; i++) {
      sb[i] = new StringBuilder(initialization);
    }

    INDArray initializationInput = encodeInput(characterIterator, initialization, numSamples);

    network.rnnClearPreviousState();
    INDArray output = network.rnnTimeStep(initializationInput);
    output = output.tensorAlongDimension((int)output.size(2) - 1, 1, 0);

    int charactersToSample = 40;
    List<Double> probs = new ArrayList<>(numSamples);
    for (int i = 0; i < charactersToSample; i++) {
      INDArray nextInput = Nd4j.zeros(numSamples, characterIterator.inputColumns());
      for (int s = 0; s < numSamples; s++) {
        double[] outputProbDistribution = new double[characterIterator.totalOutcomes()];
        for (int j = 0; j < outputProbDistribution.length; j++) {
          outputProbDistribution[j] = output.getDouble(s, j);
        }
        int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution);
        probs.add(d);
        nextInput.putScalar(new int[] {s, sampledCharacterIdx}, 1.0f);
        char c = characterIterator.convertIndexToCharacter(sampledCharacterIdx);
        if (eosChar != null && eosChar == c) {
          break;
        }
        sb[s].append(c);
      }

      output = network.rnnTimeStep(nextInput);
    }

    Map<String, Double> out = new HashMap<>();
    for (int i = 0; i < numSamples; i++) {
      out.put(sb[i].toString(), probs.get(i));
    }
    return out;
  }

  private static INDArray encodeInput(CharacterIterator characterIterator, String initialization, int numSamples) {
    INDArray initializationInput = Nd4j.zeros(numSamples, characterIterator.inputColumns(), initialization.length());
    char[] init = initialization.toCharArray();
    for (int i = 0; i < init.length; i++) {
      int idx = characterIterator.convertCharacterToIndex(init[i]);
      for (int j = 0; j < numSamples; j++) {
        initializationInput.putScalar(new int[] {j, idx, i}, 1.0f);
      }
    }
    return initializationInput;
  }

  private static int sampleFromDistribution(double[] distribution) {
    d = 0.0;
    double sum = 0.0;
    for (int t = 0; t < 10; t++) {
      d = rng.nextDouble();
      sum = 0.0;
      for (int i = 0; i < distribution.length; i++) {
        sum += distribution[i];
        if (d <= sum) {
          return i;
        }
      }
    }
    throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
  }

  public static MultiLayerConfiguration buildLSTM(int noOfHiddenLayers, int lstmLayerSize, int tbpttLength, int ioSize,
                                                  WeightInit weightInit, IUpdater updater, Activation activation) {

    NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .seed(12345)
        .l2(0.001)
        .weightInit(weightInit)
        .updater(updater)
        .list()
        .layer(0, new LSTM.Builder().nIn(ioSize).nOut(lstmLayerSize)
            .activation(activation).build());

    for (int i = 1; i < noOfHiddenLayers; i++) {
      builder = builder.layer(i, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
          .activation(activation).build());
    }
    builder.layer(noOfHiddenLayers, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize).nOut(ioSize).build())
        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength).pretrain(false).backprop(true)
        .build();
    return builder.build();
  }

  public static MultiLayerNetwork trainLSTM(int lstmLayerSize, int tbpttLength, int numEpochs, int noOfHiddenLayers,
                                            CharacterIterator iter, WeightInit weightInit, IUpdater updater, Activation activation,
                                            TrainingListener... listeners) throws IOException {
    MultiLayerConfiguration conf = buildLSTM(noOfHiddenLayers, lstmLayerSize, tbpttLength, iter.inputColumns(), weightInit, updater, activation);

    String name = lstmLayerSize + "-" + tbpttLength + "-" + numEpochs + "-" + noOfHiddenLayers + "-" + weightInit + "-" + updater + "-" + activation;

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(listeners);

    log.info("params :" + net.numParams() + ", examples: " + iter.numExamples());

    trainAndSave(numEpochs, iter, name, net);
    return net;
  }

  private static void trainAndSave(int numEpochs, CharacterIterator iter, String name, MultiLayerNetwork net) throws IOException {
    int miniBatchNumber = 0;
    int generateSamplesEveryNMinibatches = 300;
    for (int i = 0; i < numEpochs; i++) {
      while (iter.hasNext()) {
        DataSet next = iter.next();
        net.fit(next);
        if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
          String[] samples = sampleFromNetwork(net, iter, "latest trends\n", 3, '\n').keySet().toArray(new String[3]);
          for (int j = 0; j < samples.length; j++) {
            log.info("----- Sample {} -----", j);
            log.info(samples[j]);
          }
        }
      }
    }
    File locationToSave = new File("target/charLSTM-" + name + "-" + net.numParams() + "-" + iter.numExamples() + ".zip");
    assert locationToSave.createNewFile();
    ModelSerializer.writeModel(net, locationToSave, true);
  }

  public static List<String> generateInputs(String input) {
    List<String> inputs = new LinkedList<>();
    for (int i = 1; i < input.length(); i++) {
      inputs.add(input.substring(0, i));
    }
    inputs.add(input);
    return inputs;
  }

  public static <T extends SequenceElement> WeightLookupTable<T> readLookupTable(InputStream stream) throws IOException {
    WeightLookupTable<T> weightLookupTable = new InMemoryLookupTable<>();
    boolean headerRead = false;
    for (String line : IOUtils.readLines(stream, "UTF-8")) {
      String[] tokens = line.split(" ");
      if (!headerRead) {
        // reading header as "NUM_WORDS VECTOR_SIZE NUM_DOCS"
        int numWords = Integer.parseInt(tokens[0]);
        int layerSize = Integer.parseInt(tokens[1]);
        int totalNumberOfDocs = Integer.parseInt(tokens[2]);
        log.debug("Reading header - words: {}, layerSize: {}, totalNumberOfDocs: {}", numWords, layerSize, totalNumberOfDocs);
        headerRead = true;
      }

      String label = WordVectorSerializer.decodeB64(tokens[0]);
      INDArray vector = Nd4j.create(tokens.length - 1);
      if (label != null && vector != null) {
        for (int i = 1; i < tokens.length; i++) {
          vector.putScalar(i - 1, Double.parseDouble(tokens[i]));
        }
        weightLookupTable.putVector(label, vector);
      }

    }
    stream.close();
    return weightLookupTable;
  }

}
