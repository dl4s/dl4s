package com.manning.dl4s.ch7.encdec;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import javax.xml.stream.XMLStreamException;

import com.google.common.base.Joiner;
import com.manning.dl4s.ch7.TranslatorTool;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * a {@link TranslatorTool} based on encoder decoder neural network.
 */
public class MTNetwork implements TranslatorTool {

  private final Logger log = LoggerFactory.getLogger(getClass());

  private static final double LEARNING_RATE = 0.0001;
  private static final int TBPTT_SIZE = 25;
  private static final int EMBEDDING_WIDTH = 100;
  private static final int HIDDEN_LAYER_WIDTH = 300;
  static final int ROW_SIZE = 40;
  private static final int MINIBATCH_SIZE = 100;

  private static final String MODEL_FILENAME = "mt_lstm_train.zip"; // filename of the model
  private static final String BACKUP_MODEL_FILENAME = "mt_lstm_train.zip"; // filename of the model
  private static final long TEST_EACH_MS = TimeUnit.MINUTES.toMillis(2); // test the model with this period
  private static final long SAVE_EACH_MS = TimeUnit.MINUTES.toMillis(10); // save the model with this period
  private static final Random rnd = new Random(new Date().getTime());
  private static final int EPOCHS = 1;

  private final ComputationGraph net;

  private ParallelCorpusProcessor corpusProcessor;


  public MTNetwork(String tmxFile, String sourceCode, String targetCode, TrainingListener... listeners) {

    try {
      corpusProcessor = new ParallelCorpusProcessor(tmxFile, sourceCode, targetCode, true);
      corpusProcessor.process();

      Map<String, Double> dict = corpusProcessor.getDict();

      File networkFile = new File(toTempPath(MODEL_FILENAME));
      if (networkFile.exists()) {
        log.info("Loading the existing network from {}", networkFile);
        net = ModelSerializer.restoreComputationGraph(networkFile);

      } else {
        log.info("Creating a new network...");
        net = createComputationGraph(dict.size());
      }

      net.init();
      net.setListeners(listeners);
      train();
    } catch (Throwable t) {
      throw new RuntimeException(t);
    }
  }

  public MTNetwork(File modelFile, boolean train, String tmxFile, String sourceCode, String targetCode, StatsListener... listeners) throws IOException, XMLStreamException {
    corpusProcessor = new ParallelCorpusProcessor(tmxFile, sourceCode, targetCode, true);
    corpusProcessor.process();
    net = ModelSerializer.restoreComputationGraph(modelFile);
    net.init();
    net.setListeners(listeners);
    if (train) {
      train();
    }
  }

  private ComputationGraph createComputationGraph(int dictionarySize) {
    final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
        .updater(new Sgd(LEARNING_RATE))
        .weightInit(WeightInit.XAVIER)
        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

    final ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder()

        .pretrain(false)
        .backprop(true)
        .backpropType(BackpropType.TruncatedBPTT)
        .tBPTTBackwardLength(TBPTT_SIZE)
        .tBPTTForwardLength(TBPTT_SIZE)
        .addInputs("inputLine", "decoderInput")
        .setInputTypes(InputType.recurrent(dictionarySize), InputType.recurrent(dictionarySize))
        .addLayer("embeddingEncoder",
            new EmbeddingLayer.Builder()
                .nIn(dictionarySize)
                .nOut(EMBEDDING_WIDTH)
                .build(),
            "inputLine")
        .addLayer("encoder",
            new LSTM.Builder()
                .nIn(EMBEDDING_WIDTH)
                .nOut(HIDDEN_LAYER_WIDTH)
                .activation(Activation.TANH)
                .gateActivationFunction(Activation.HARDSIGMOID)
                .build(),
            "embeddingEncoder")
        .addLayer("encoder2",
            new LSTM.Builder()
                .nIn(HIDDEN_LAYER_WIDTH)
                .nOut(HIDDEN_LAYER_WIDTH)
                .activation(Activation.TANH)
                .build(),
            "encoder")
        .addLayer("encoder3",
            new LSTM.Builder()
                .nIn(HIDDEN_LAYER_WIDTH)
                .nOut(HIDDEN_LAYER_WIDTH)
                .activation(Activation.TANH)
                .build(),
            "encoder2")
        .addVertex("thoughtVector", new LastTimeStepVertex("inputLine"), "encoder3")
        .addVertex("dup", new DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
        .addVertex("merge", new MergeVertex(), "decoderInput", "dup")
        .addLayer("decoder",
            new LSTM.Builder()
                .nIn(dictionarySize + HIDDEN_LAYER_WIDTH)
                .nOut(HIDDEN_LAYER_WIDTH)
                .activation(Activation.TANH)
                .gateActivationFunction(Activation.HARDSIGMOID)
                .build(),
            "merge")
        .addLayer("decoder2",
            new LSTM.Builder()
                .nIn(HIDDEN_LAYER_WIDTH)
                .nOut(HIDDEN_LAYER_WIDTH)
                .activation(Activation.TANH)
                .build(),
            "decoder")
        .addLayer("decoder3",
            new LSTM.Builder()
                .nIn(HIDDEN_LAYER_WIDTH)
                .nOut(HIDDEN_LAYER_WIDTH)
                .activation(Activation.TANH)
                .build(),
            "decoder2")
        .addLayer("output",
            new RnnOutputLayer.Builder()
                .nIn(HIDDEN_LAYER_WIDTH)
                .nOut(dictionarySize)
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .build(),
            "decoder3")
        .setOutputs("output");

    return new ComputationGraph(graphBuilder.build());
  }

  private void train() throws IOException {
    File networkFile = new File(toTempPath(MODEL_FILENAME));
    long lastTestTime = System.currentTimeMillis();
    long lastSaveTime = System.currentTimeMillis();
    ParallelCorpusIterator parallelCorpusIterator = new ParallelCorpusIterator(corpusProcessor, MINIBATCH_SIZE, ROW_SIZE);
    for (int epoch = 1; epoch < EPOCHS; ++epoch) {
      log.info("Epoch {}", epoch);
      int lastPerc = 0;
      while (parallelCorpusIterator.hasNext()) {
        net.rnnClearPreviousState();
        MultiDataSet multiDataSet = parallelCorpusIterator.next();
        net.fit(multiDataSet);

        int newPerc = (parallelCorpusIterator.batch() * 100 / parallelCorpusIterator.totalBatches());
        if (newPerc != lastPerc) {
          log.info("Epoch complete: {}%", newPerc);
          lastPerc = newPerc;
        }
        if (System.currentTimeMillis() - lastTestTime > TEST_EACH_MS) {
          test();
          lastTestTime = System.currentTimeMillis();
        }
        if (System.currentTimeMillis() - lastSaveTime > SAVE_EACH_MS) {
          saveModel(networkFile);
          lastSaveTime = System.currentTimeMillis();
        }
      }
    }
  }

  private void test() {
    log.info("======================== TEST ========================");
    int selected = rnd.nextInt(corpusProcessor.getCorpus().size() - 1);
    if (selected % 2 == 1) {
      selected++;
    }
    List<Double> rowIn = new ArrayList<>(corpusProcessor.getCorpus().get(selected));
    log.info("In: ");
    StringBuilder builder = new StringBuilder();
    for (Double idx : rowIn) {
      builder.append(corpusProcessor.getRevDict().get(idx)).append(" ");
    }
    log.info(builder.toString());
    log.info("Out: ");
    log.info("{}", output(rowIn, true, 0));
    log.info("====================== TEST END ======================");
  }

  public Collection<String> output(String query, double score) {
    Collection<String> tokens = new LinkedList<>();
    corpusProcessor.tokenizeLine(query, tokens);
    List<Double> doubles = corpusProcessor.wordsToIndexes(tokens);
    return output(doubles, false, score);
  }

  private Collection<String> output(List<Double> rowIn, boolean printUnknowns, double score) {
    Collection<String> result = new LinkedList<>();

    Collections.reverse(rowIn);
    INDArray in = Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0])), new int[] { 1, 1, rowIn.size() });
    Map<String, Double> dict = corpusProcessor.getDict();
    Map<Double, String> revDict = corpusProcessor.getRevDict();
    double[] decodeArr = new double[dict.size()];
    decodeArr[2] = 1;
    INDArray decode = Nd4j.create(decodeArr, new int[] { 1, dict.size(), 1 });
    net.feedForward(new INDArray[] { in, decode }, false, false);
    org.deeplearning4j.nn.layers.recurrent.LSTM decoder = (org.deeplearning4j.nn.layers.recurrent.LSTM) net
            .getLayer("decoder");
    Layer output = net.getLayer("output");
    GraphVertex mergeVertex = net.getVertex("merge");
    INDArray thoughtVector = mergeVertex.getInputs()[1];
    LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
    for (int row = 0; row < rowIn.size(); ++row) {
      mergeVertex.setInputs(decode, thoughtVector);
      INDArray merged = mergeVertex.doForward(false, mgr);
      INDArray activateDec = decoder.rnnTimeStep(merged, mgr);
      INDArray out = output.activate(activateDec, false, mgr);
      double d = rnd.nextDouble();
      double sum = 0.0;
      int idx = -1;
      for (int s = 0; s < out.size(1); s++) {
        sum += out.getDouble(0, s, 0);
        if (d <= sum) {
          idx = s;
          if (printUnknowns || s != 0) {
            String s1 = revDict.get((double) s);
            result.add(s1);
          }
          break;
        }
      }
      if (idx == 1) {
        break;
      }
      if (idx >= 0) {
        double[] newDecodeArr = new double[dict.size()];
        newDecodeArr[idx] = 1;
        decode = Nd4j.create(newDecodeArr, new int[]{1, dict.size(), 1});
      }
    }
    return result;
  }

  private void saveModel(File networkFile) throws IOException {
    log.info("Saving the model...");
    File backup = new File(toTempPath(BACKUP_MODEL_FILENAME));
    if (networkFile.exists()) {
      if (backup.exists()) {
        backup.delete();
      }
      networkFile.renameTo(backup);
    }
    ModelSerializer.writeModel(net, networkFile, true);
    log.info("Model saved to {}", networkFile.getAbsolutePath());
  }

  private String toTempPath(String path) {
    return System.getProperty("java.io.tmpdir") + "/" + path;
  }

  @Override
  public Collection<Translation> translate(String text) {
    double score = 0d;
    String string = Joiner.on(' ').join(output(text, score));
    Translation translation = new Translation(string, score);
    return Collections.singletonList(translation);
  }
}
