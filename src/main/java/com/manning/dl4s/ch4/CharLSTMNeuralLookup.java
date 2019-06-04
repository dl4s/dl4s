package com.manning.dl4s.ch4;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.manning.dl4s.utils.CharacterIterator;
import com.manning.dl4s.utils.NeuralNetworksUtils;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.search.suggest.Lookup;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * a {@link Lookup} using a char LSTM network for generating suggestions
 */
public class CharLSTMNeuralLookup extends Lookup {

  private int lstmLayerSize;
  private int miniBatchSize;
  private int exampleLength;
  private int tbpttLength;
  private int numEpochs;
  private int noOfHiddenLayers;
  private WeightInit weightInit;
  private IUpdater updater;
  private Activation activation;

  protected CharacterIterator characterIterator;
  protected MultiLayerNetwork network;

  public CharLSTMNeuralLookup(MultiLayerNetwork net, CharacterIterator iter) {
    network = net;
    characterIterator = iter;
  }

  public CharLSTMNeuralLookup(int lstmLayerSize, int miniBatchSize, int exampleLength, int tbpttLength,
                              int numEpochs, int noOfHiddenLayers, WeightInit weightInit,
                              IUpdater updater, Activation activation) {
    this.lstmLayerSize = lstmLayerSize;
    this.miniBatchSize = miniBatchSize;
    this.exampleLength = exampleLength;
    this.tbpttLength = tbpttLength;
    this.numEpochs = numEpochs;
    this.noOfHiddenLayers = noOfHiddenLayers;
    this.weightInit = weightInit;
    this.updater = updater;
    this.activation = activation;
  }

  @Override
  public long getCount() {
    return -1L;
  }

  @Override
  public void build(InputIterator inputIterator) throws IOException {
    Path tempFile = Files.createTempFile("chars",".txt");
    FileOutputStream outputStream = new FileOutputStream(tempFile.toFile());
    for (BytesRef surfaceForm; (surfaceForm = inputIterator.next()) != null;) {
      outputStream.write(surfaceForm.bytes);
      outputStream.write("\n".getBytes());
    }
    outputStream.flush();
    outputStream.close();
    characterIterator = new CharacterIterator(tempFile.toAbsolutePath().toString(),
        Charset.defaultCharset(), miniBatchSize, exampleLength);
    this.network = NeuralNetworksUtils.trainLSTM(lstmLayerSize, tbpttLength, numEpochs, noOfHiddenLayers,
        characterIterator, weightInit, updater, activation, new ScoreIterationListener(1000));
  }

  @Override
  public List<LookupResult> lookup(CharSequence key, Set<BytesRef> contexts, boolean onlyMorePopular, int num) throws IOException {
    List<LookupResult> results = new LinkedList<>();
    Map<String, Double> output = NeuralNetworksUtils.sampleFromNetwork(network, characterIterator, key.toString(), num, null);
    for (Map.Entry<String, Double> entry : output.entrySet()) {
      results.add(new LookupResult(entry.getKey(), entry.getValue().longValue()));
    }
    return results;
  }



  @Override
  public boolean store(DataOutput output) throws IOException {
    // TODO : implement this
    return false;
  }

  @Override
  public boolean load(DataInput input) throws IOException {
    // TODO : implement this
    return false;
  }

  @Override
  public long ramBytesUsed() {
    // TODO : implement this
    return 0;
  }
}
