package com.manning.dl4s.ch3;

import com.manning.dl4s.utils.CharacterIterator;
import com.manning.dl4s.utils.NeuralNetworksUtils;
import com.manning.dl4s.utils.encdec.EncoderDecoderLSTM;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.*;

import java.io.File;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;

import static org.junit.Assert.assertNotNull;

/**
 * Tests for {@link AltQueriesQueryParser}
 */
public class AltQueriesQueryParserTest {

  @Test
  public void testCharLSTMParseExpansionWithTraining() throws Exception {

    int lstmLayerSize = 60;
    int miniBatchSize = 1000;
    int exampleLength = 60;
    int tbpttLength = 20;
    int numEpochs = 1;
    int noOfHiddenLayers = 1;
    double learningRate = 0.01;

    String file = getClass().getResource("/queries2.shuf.txt").getFile();
    CharacterIterator iter = new CharacterIterator(file, Charset.forName("ISO-8859-1"), miniBatchSize, exampleLength, CharacterIterator.getMinimalCharacterSet(), new Random(12345));

    IUpdater updater = RmsProp.builder().learningRate(learningRate).build();

    MultiLayerNetwork net = NeuralNetworksUtils.trainLSTM(lstmLayerSize, tbpttLength, numEpochs, noOfHiddenLayers, iter,
            WeightInit.NORMAL, updater, Activation.TANH);

    AltQueriesQueryParser altQueriesQueryParser = new AltQueriesQueryParser("text", new EnglishAnalyzer(null), net, iter);

    String[] queries = new String[] {"latest trends", "covfefe", "concerts", "music events"};
    for (String query : queries) {
      assertNotNull(altQueriesQueryParser.parse(query));
    }

  }

  @Test
  public void testRestoredCharLSTMParseExpansion() throws Exception {

    int miniBatchSize = 1000;
    int exampleLength = 60;

    String file = getClass().getResource("/queries2.shuf.txt").getFile();
    CharacterIterator iter = new CharacterIterator(file, Charset.forName("ISO-8859-1"), miniBatchSize, exampleLength, CharacterIterator.getMinimalCharacterSet(), new Random(12345));

    MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork("src/test/resources/models/ch3/charLSTM-120-60-3-2-0.1-XAVIER-RMSPROP-SOFTSIGN-220037-235740.zip");

    AltQueriesQueryParser altQueriesQueryParser = new AltQueriesQueryParser("text", new EnglishAnalyzer(null), net, iter);

    String[] queries = new String[] {"latest trends", "covfefe", "concerts", "music events"};
    for (String query : queries) {
      assertNotNull(altQueriesQueryParser.parse(query));
    }

  }

  @Test
  public void testSeq2SeqParseExpansion() throws Exception {

    URL resource = getClass().getResource("/queries2.shuf.txt");
    EncoderDecoderLSTM encDecLSTM = new EncoderDecoderLSTM(new File("target/encdec" + System.currentTimeMillis() + ".zip"), new File(resource.toURI()));
    Seq2SeqAltQueriesQueryParser seq2SeqAltQueriesQueryParser = new Seq2SeqAltQueriesQueryParser("text", new EnglishAnalyzer(), encDecLSTM);

    String[] queries = new String[] {"latest trends", "covfefe", "concerts", "music events"};
    for (String query : queries) {
      assertNotNull(seq2SeqAltQueriesQueryParser.parse(query));
    }

  }

}