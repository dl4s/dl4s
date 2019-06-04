package com.manning.dl4s.utils.encdec;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Slightly adapted version of DL4J's {@link EncoderDecoderLSTM}
 *
 * @see <a href="https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/encdec/EncoderDecoderLSTM.java"></a>
 */
public class EncoderDecoderLSTM {

  private final Logger log = LoggerFactory.getLogger(getClass());

  /**
   * Dictionary that maps words into numbers.
   */
  private final Map<String, Double> dict = new HashMap<>();

  /**
   * Reverse map of {@link #dict}.
   */
  private final Map<Double, String> revDict = new HashMap<>();

  /**
   * The contents of the corpus. This is a list of sentences (each word of the
   * sentence is denoted by a {@link Double}).
   */
  private final List<List<Double>> corpus = new ArrayList<>();

  private static final int HIDDEN_LAYER_WIDTH = 300;
  private static final int EMBEDDING_WIDTH = 100;
  private static final String MODEL_FILENAME = "encdec_train.zip"; // filename of the model
  private static final String BACKUP_MODEL_FILENAME = "encdec_train.bak.zip"; // filename of the previous version of the model (backup)
  private static final int MINIBATCH_SIZE = 32;
  private static final Random rnd = new Random(new Date().getTime());
  private static final long SAVE_EACH_MS = TimeUnit.MINUTES.toMillis(5); // save the model with this period
  private static final long TEST_EACH_MS = TimeUnit.MINUTES.toMillis(3); // test the model with this period
  private static final int MAX_DICT = 20000; // this number of most frequent words will be used
  private static final int TBPTT_SIZE = 25;
  private static final int ROW_SIZE = 40; // maximum line length in tokens
  private static final int EPOCHS = 3;

  /**
   * The delay between invocations of {@link System#gc()} in
   * milliseconds. If VRAM is being exhausted, reduce this value. Increase
   * this value to yield better performance.
   */
  private static final int GC_WINDOW = 2000;

  private static final int MACROBATCH_SIZE = 20; // see CorpusIterator

  private String corpusFilename;

  private ComputationGraph net;
  private CorpusProcessor corpusProcessor;

  public EncoderDecoderLSTM(File modelFile, File corpusFile, TrainingListener... listeners) throws IOException {
    this.corpusFilename = corpusFile.getAbsolutePath();
    Nd4j.getMemoryManager().setAutoGcWindow(GC_WINDOW);
    if (modelFile.exists()) {
      net = ModelSerializer.restoreComputationGraph(modelFile);
    } else {
      train(listeners);
    }
  }

  private void train(TrainingListener... listeners) throws IOException {
    Nd4j.getMemoryManager().setAutoGcWindow(GC_WINDOW);

    createDictionary();

    File networkFile = new File(toTempPath(MODEL_FILENAME));
    int offset = 0;
    if (networkFile.exists()) {
      log.info("Loading the existing network...");
      net = ModelSerializer.restoreComputationGraph(networkFile);
    } else {
      log.info("Creating a new network...");
      createComputationGraph();
    }
    log.info("Number of parameters {}", net.numParams());
    if (listeners != null && listeners.length > 0) {
      net.setListeners(listeners);
    }
    train(networkFile, offset);
  }

  /**
   * Configure and initialize the computation graph. This is done once in the
   * beginning to prepare the {@link #net} for training.
   */
  private void createComputationGraph() {
    final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .miniBatch(true)
        .updater(Updater.RMSPROP)
        .weightInit(WeightInit.XAVIER)
        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

    final GraphBuilder graphBuilder = builder.graphBuilder()
        .pretrain(false)
        .backprop(true)
        .backpropType(BackpropType.Standard)
        .tBPTTBackwardLength(TBPTT_SIZE)
        .tBPTTForwardLength(TBPTT_SIZE)
        .addInputs("inputLine", "decoderInput")
        .setInputTypes(InputType.recurrent(dict.size()), InputType.recurrent(dict.size()))
        .addLayer("embeddingEncoder",
            new EmbeddingLayer.Builder()
                .nIn(dict.size())
                .nOut(EMBEDDING_WIDTH)
                .build(),
            "inputLine")
        .addLayer("encoder",
            new LSTM.Builder()
                .nIn(EMBEDDING_WIDTH)
                .nOut(HIDDEN_LAYER_WIDTH)
                .activation(Activation.TANH)
                .build(),
            "embeddingEncoder")
        .addVertex("thoughtVector", new LastTimeStepVertex("inputLine"), "encoder")
        .addVertex("dup", new DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
        .addVertex("merge", new MergeVertex(), "decoderInput", "dup")
        .addLayer("decoder",
            new LSTM.Builder()
                .nIn(dict.size() + HIDDEN_LAYER_WIDTH)
                .nOut(HIDDEN_LAYER_WIDTH)
                .activation(Activation.TANH)
                .build(),
            "merge")
        .addLayer("output",
            new RnnOutputLayer.Builder()
                .nIn(HIDDEN_LAYER_WIDTH)
                .nOut(dict.size())
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .build(),
            "decoder")
        .setOutputs("output");

    net = new ComputationGraph(graphBuilder.build());
    net.init();
  }

  private void train(File networkFile, int offset) throws IOException {
    long lastSaveTime = System.currentTimeMillis();
    long lastTestTime = System.currentTimeMillis();
    CorpusIterator logsIterator = new CorpusIterator(corpus, MINIBATCH_SIZE, MACROBATCH_SIZE, dict.size(), ROW_SIZE);
    for (int epoch = 1; epoch < EPOCHS; ++epoch) {
      log.info("Epoch {}", epoch);
      if (epoch == 1) {
        logsIterator.setCurrentBatch(offset);
      } else {
        logsIterator.reset();
      }
      int lastPerc = 0;
      while (logsIterator.hasNext()) {
        MultiDataSet multiDataSet = logsIterator.next();
        net.fit(multiDataSet);

        logsIterator.nextMacroBatch();
        log.info("Batch = {}", logsIterator.batch());
        int newPerc = (logsIterator.batch() * 100 / logsIterator.totalBatches());
        if (newPerc != lastPerc) {
          log.info("Epoch complete: {}%", newPerc);
          lastPerc = newPerc;
        }
        if (System.currentTimeMillis() - lastSaveTime > SAVE_EACH_MS) {
          saveModel(networkFile);
          lastSaveTime = System.currentTimeMillis();
        }
        if (System.currentTimeMillis() - lastTestTime > TEST_EACH_MS) {
          test();
          lastTestTime = System.currentTimeMillis();
        }
      }
    }
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
    log.info("Done.");
  }

  private void test() {
    log.info("======================== TEST ========================");
    int selected = rnd.nextInt(corpus.size());
    List<Double> rowIn = new ArrayList<>(corpus.get(selected));
    log.info("In: ");
    for (Double idx : rowIn) {
      System.out.print(revDict.get(idx) + " ");
    }
    System.out.println();
    log.info("Out: ");
    output(rowIn, true);
    log.info("====================== TEST END ======================");
  }

  public Collection<String> output(String query) {
    Collection<String> tokens = new LinkedList<>();
    corpusProcessor.tokenizeLine(query, tokens, false);
    List<Double> doubles = corpusProcessor.wordsToIndexes(tokens);
    return output(doubles, false);
  }

  public INDArray getThoughtVector(String... words) {
    List<Double> rowIn = corpusProcessor.wordsToIndexes(Arrays.asList(words));

    net.rnnClearPreviousState();
    Collections.reverse(rowIn);
    INDArray in = Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0])), new int[] { 1, 1, rowIn.size() });
    double[] decodeArr = new double[dict.size()];
    decodeArr[2] = 1;
    INDArray decode = Nd4j.create(decodeArr, new int[] { 1, dict.size(), 1 });
    net.feedForward(new INDArray[] { in, decode }, false, false);
    return net.getVertex("merge").getInputs()[1].tensorAlongDimension(0, 0, 1);
  }

  public INDArray getThoughtVector(INDArray in, boolean train) {
    net.feedForward(in, train);
    return net.getVertex("merge").getInputs()[1];
  }

  private Collection<String> output(List<Double> rowIn, boolean printUnknowns) {
    Collection<String> result = new LinkedList<>();

    net.rnnClearPreviousState();
    Collections.reverse(rowIn);
    INDArray in = Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0])), new int[] { 1, 1, rowIn.size() });
    double[] decodeArr = new double[dict.size()];
    decodeArr[2] = 1;
    INDArray decode = Nd4j.create(decodeArr, new int[] { 1, dict.size(), 1 });
    net.feedForward(new INDArray[] { in, decode }, false, false);
    org.deeplearning4j.nn.layers.recurrent.LSTM decoder = (org.deeplearning4j.nn.layers.recurrent.LSTM) net
        .getLayer("decoder");
    Layer output = net.getLayer("output");
    GraphVertex mergeVertex = net.getVertex("merge");
    INDArray[] inputs = mergeVertex.getInputs();
    INDArray thoughtVector = inputs[1];
    LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
    for (int row = 0; row < ROW_SIZE; ++row) {
      mergeVertex.setInputs(decode, thoughtVector);
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
            String s1 = revDict.get((double) s) + " ";
            result.add(s1);
          }
          break;
        }
      }
      if (idx == 1) {
        break;
      }
      double[] newDecodeArr = new double[dict.size()];
      newDecodeArr[idx] = 1;
      decode = Nd4j.create(newDecodeArr, new int[] {1, dict.size(), 1});
    }
    return result;
  }

  private void createDictionary() throws IOException, FileNotFoundException {
    double idx = 3.0;
    dict.put("<unk>", 0.0);
    revDict.put(0.0, "<unk>");
    dict.put("<eos>", 1.0);
    revDict.put(1.0, "<eos>");
    dict.put("<go>", 2.0);
    revDict.put(2.0, "<go>");
    String CHARS = "-\\/_&" + CorpusProcessor.SPECIALS;
    for (char c : CHARS.toCharArray()) {
      if (!dict.containsKey(c)) {
        dict.put(String.valueOf(c), idx);
        revDict.put(idx, String.valueOf(c));
        ++idx;
      }
    }
    log.info("Building the dictionary...");
    corpusProcessor = new CorpusProcessor(corpusFilename, ROW_SIZE, true);
    corpusProcessor.start();
    Map<String, Double> freqs = corpusProcessor.getFreq();
    Map<Double, Set<String>> freqMap = new TreeMap<>((o1, o2) -> (int) (o2 - o1)); // tokens of the same frequency fall under the same key, the order is reversed so the most frequent tokens go first
    for (Entry<String, Double> entry : freqs.entrySet()) {
      Set<String> set = freqMap.computeIfAbsent(entry.getValue(), k -> new TreeSet<>());
      // tokens of the same frequency would be sorted alphabetically
      set.add(entry.getKey());
    }
    int cnt = 0;
    // the tokens order is preserved for TreeSet
    Set<String> dictSet = new TreeSet<>(dict.keySet());
    // get most frequent tokens and put them to dictSet
    for (Entry<Double, Set<String>> entry : freqMap.entrySet()) {
      for (String val : entry.getValue()) {
        if (dictSet.add(val) && ++cnt >= MAX_DICT) {
          break;
        }
      }
      if (cnt >= MAX_DICT) {
        break;
      }
    }
    // all of the above means that the dictionary with the same MAX_DICT constraint and made from the same source file will always be
    // the same, the tokens always correspond to the same number so we don't need to save/restore the dictionary
    log.info("Dictionary is ready, size is {}", dictSet.size());
    // index the dictionary and build the reverse dictionary for lookups
    for (String word : dictSet) {
      if (!dict.containsKey(word)) {
        dict.put(word, idx);
        revDict.put(idx, word);
        ++idx;
      }
    }
    log.info("Total dictionary size is {}. Processing the dataset...", dict.size());
    corpusProcessor = new CorpusProcessor(corpusFilename, ROW_SIZE, false) {
      @Override
      public void processLine(String lastLine) {
        List<String> words = new ArrayList<>();
        tokenizeLine(lastLine, words, true);
        corpus.add(wordsToIndexes(words));
      }
    };
    corpusProcessor.setDict(dict);
    corpusProcessor.start();
    log.info("Done. Corpus size is {}", corpus.size());
  }

  private String toTempPath(String path) {
//        return System.getProperty("java.io.tmpdir") + "/" + path;
    return "/tmp/" + path;
  }


}
