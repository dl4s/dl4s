package com.manning.dl4s.ch6;

import java.util.List;
import java.util.Map;

import com.google.common.base.Joiner;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Listener that extracts thought vectors from a seq2seq model and stores them in a {@link WeightLookupTable}
 */
public class ThoughtVectorsListener implements TrainingListener {

  private final WeightLookupTable lookupTable;
  private final String hiddenLayerName;
  private final Map<Double, String> revDict;
  private final String inputLayerName;

  public ThoughtVectorsListener(WeightLookupTable lookupTable, String inputLayerName, String hiddenLayerName, Map<Double, String> revDict) {
    this.lookupTable = lookupTable;
    this.inputLayerName = inputLayerName;
    this.hiddenLayerName = hiddenLayerName;
    this.revDict = revDict;
  }

  @Override
  public void onEpochStart(Model model) {

  }

  @Override
  public void onEpochEnd(Model model) {

  }

  @Override
  public void onForwardPass(Model model, List<INDArray> activations) {

  }

  @Override
  public void onForwardPass(Model model, Map<String, INDArray> activations) {
    extractThoughtVector(activations);
  }

  @Override
  public void onGradientCalculation(Model model) {

  }

  @Override
  public void onBackwardPass(Model model) {

  }

  private void extractThoughtVector(Map<String, INDArray> activations) {
    INDArray input = activations.get(inputLayerName);
    INDArray thoughtVector = activations.get(hiddenLayerName);
    for (int i = 0; i < input.size(0); i++) {
      for (int j = 0; j < input.size(1); j++) {
        int size = (int)input.size(2);
        String[] words = new String[size];
        for (int s = 0; s < size; s++) {
          words[s] = revDict.get(input.getDouble(i, j, s)) + " ";
        }
        String sequence = Joiner.on(' ').join(words);
        lookupTable.putVector(sequence, thoughtVector.tensorAlongDimension(i, j));
      }
    }
  }

  @Override
  public void iterationDone(Model model, int i, int i1) {

  }
}
