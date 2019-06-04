package com.manning.dl4s.ch7.encdec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@link MultiDataSetIterator} to iterate over a parallel corpus.
 */
@SuppressWarnings("serial")
public class ParallelCorpusIterator implements MultiDataSetIterator {

  private final Logger log = LoggerFactory.getLogger(getClass());

  private ParallelCorpusProcessor corpusProcessor;
  private int batchSize;
  private int totalBatches;
  private int currentBatch = 0;
  private int dictSize;
  private int rowSize;


  public ParallelCorpusIterator(ParallelCorpusProcessor corpusProcessor, int minibatchSize, int rowSize) {
    this.corpusProcessor = corpusProcessor;
    this.batchSize = minibatchSize;
    this.dictSize = corpusProcessor.getDict().size();
    this.rowSize = rowSize;
    totalBatches = (int) Math.ceil((double) corpusProcessor.getCorpus().size() / batchSize);
  }

  @Override
  public boolean hasNext() {
    return currentBatch < totalBatches;
  }

  @Override
  public MultiDataSet next() {
    return next(batchSize);
  }

  @Override
  public MultiDataSet next(int num) {
    int i = currentBatch * batchSize;
    List<List<Double>> corpus = corpusProcessor.getCorpus();
    int currentBatchSize = Math.min(batchSize, corpus.size() - i - 1);
    INDArray input = Nd4j.zeros(currentBatchSize, 1, rowSize);
    INDArray prediction = Nd4j.zeros(currentBatchSize, dictSize, rowSize);
    INDArray decode = Nd4j.zeros(currentBatchSize, dictSize, rowSize);
    INDArray inputMask = Nd4j.zeros(currentBatchSize, rowSize);
    // this mask is also used for the decoder input, the length is the same
    INDArray predictionMask = Nd4j.zeros(currentBatchSize, rowSize);
    try {
      for (int j = 0; j < currentBatchSize; j++) {
        List<Double> source = corpus.get(i);
        List<Double> rowIn = new ArrayList<>(source);
        Collections.reverse(rowIn);
        List<Double> target = corpus.get(i + 1);
        List<Double> rowPred = new ArrayList<>(target);
        rowPred.add(corpusProcessor.getDict().get(ParallelCorpusProcessor.EOS)); // add <eos> token
        // replace the entire row in "input" using NDArrayIndex, it's faster than putScalar(); input is NOT made of one-hot vectors
        // because of the embedding layer that accepts token indexes directly
        input.put(new INDArrayIndex[] {NDArrayIndex.point(j), NDArrayIndex.point(0), NDArrayIndex.interval(0, rowIn.size())},
            Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0]))));
        inputMask.put(new INDArrayIndex[] {NDArrayIndex.point(j), NDArrayIndex.interval(0, rowIn.size())}, Nd4j.ones(rowIn.size()));
        predictionMask.put(new INDArrayIndex[] {NDArrayIndex.point(j), NDArrayIndex.interval(0, rowPred.size())},
            Nd4j.ones(rowPred.size()));
        // prediction (output) and decode ARE one-hots though, I couldn't add an embedding layer on top of the decoder and I'm not sure
        // it's a good idea either
        double predOneHot[][] = new double[dictSize][rowPred.size()];
        double decodeOneHot[][] = new double[dictSize][rowPred.size()];
        decodeOneHot[corpusProcessor.getDict().get(ParallelCorpusProcessor.GO).intValue()][0] = 1; // <go> token
        int predIdx = 0;
        for (Double pred : rowPred) {
          predOneHot[pred.intValue()][predIdx] = 1;
          if (predIdx < rowPred.size() - 1) { // put the same vals to decode with +1 offset except the last token that is <eos>
            decodeOneHot[pred.intValue()][predIdx + 1] = 1;
          }
          predIdx++;
        }
        prediction.put(new INDArrayIndex[] {NDArrayIndex.point(j), NDArrayIndex.interval(0, dictSize),
            NDArrayIndex.interval(0, rowPred.size())}, Nd4j.create(predOneHot));
        decode.put(new INDArrayIndex[] {NDArrayIndex.point(j), NDArrayIndex.interval(0, dictSize),
            NDArrayIndex.interval(0, rowPred.size())}, Nd4j.create(decodeOneHot));
        i+=2;
      }
      currentBatch += 1;
    } catch (Exception ex) {
      log.error("error while iterating: {}", ex.getLocalizedMessage());
    }
    return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] {input, decode}, new INDArray[] {prediction},
        new INDArray[] {inputMask, predictionMask}, new INDArray[] {predictionMask});
  }

  @Override
  public MultiDataSetPreProcessor getPreProcessor() {
    return null;
  }

  @Override
  public boolean resetSupported() {
    // we don't want this iterator to be reset on each macrobatch pseudo-epoch
    return false;
  }

  @Override
  public boolean asyncSupported() {
    return true;
  }

  @Override
  public void reset() {
    // but we still can do it manually before the epoch starts
    currentBatch = 0;
  }

  public int batch() {
    return currentBatch;
  }

  public int totalBatches() {
    return totalBatches;
  }

  public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
  }
}
