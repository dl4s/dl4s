package com.manning.dl4s.ch4;

import com.manning.dl4s.utils.CharacterIterator;
import com.manning.dl4s.utils.NeuralNetworksUtils;
import org.apache.commons.io.IOUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.spell.Dictionary;
import org.apache.lucene.search.spell.LuceneDictionary;
import org.apache.lucene.search.suggest.DocumentDictionary;
import org.apache.lucene.search.suggest.FileDictionary;
import org.apache.lucene.search.suggest.Lookup;
import org.apache.lucene.search.suggest.analyzing.AnalyzingInfixSuggester;
import org.apache.lucene.search.suggest.analyzing.AnalyzingSuggester;
import org.apache.lucene.search.suggest.analyzing.FreeTextSuggester;
import org.apache.lucene.search.suggest.jaspell.JaspellLookup;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.RmsProp;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;

/**
 * Tests for suggesters
 */
public class LookupTest {

  @Test
  public void testJaspellSuggesterFromQueries() throws Exception {
    Lookup lookup = new JaspellLookup();
    Path path = Paths.get("src/test/resources/queries2.shuf.txt");
    Dictionary dictionary = new FileDictionary(new FileInputStream(path.toFile()));
    lookup.build(dictionary);

    List<String> inputs = NeuralNetworksUtils.generateInputs("music is my aircraft");
    generateSuggestions(lookup, inputs);
  }

  @Test
  public void testAnalyzingSuggesterFromQueries() throws Exception {
    Directory dir = new RAMDirectory();
    try {
      Analyzer indexTimeAnalyzer = new StandardAnalyzer();
      Analyzer queryTimeAnalyzer = new StandardAnalyzer();
      AnalyzingSuggester lookup = new AnalyzingSuggester(dir, "target/analizerlookup", indexTimeAnalyzer, queryTimeAnalyzer);
      Path path = Paths.get("src/test/resources/queries2.shuf.txt");
      Dictionary dictionary = new FileDictionary(new FileInputStream(path.toFile()));
      lookup.build(dictionary);

      List<String> inputs = NeuralNetworksUtils.generateInputs("music is my aircraft");
      generateSuggestions(lookup, inputs);
    } finally {
      dir.close();
    }
  }

  @Test
  public void testAnalyzingInfixSuggesterFromQueries() throws Exception {
    Directory dir = new RAMDirectory();
    try {
      Analyzer buildTimeAnalyzer = new StandardAnalyzer();
      Analyzer lookupTimeAnalyzer = new StandardAnalyzer();
      AnalyzingInfixSuggester lookup = new AnalyzingInfixSuggester(dir, buildTimeAnalyzer, lookupTimeAnalyzer,
              AnalyzingSuggester.PRESERVE_SEP, true, true, false);
      Path path = Paths.get("src/test/resources/queries2.shuf.txt");
      Dictionary dictionary = new FileDictionary(new FileInputStream(path.toFile()));
      lookup.build(dictionary);

      List<String> inputs = NeuralNetworksUtils.generateInputs("music is my aircraft");
      generateSuggestions(lookup, inputs);
      lookup.close();
    } finally {
      dir.close();
    }
  }

  @Test
  public void testFreeTextSuggesterFromQueries() throws Exception {
    Lookup lookup = new FreeTextSuggester(new StandardAnalyzer());
    Path path = Paths.get("src/test/resources/queries2.shuf.txt");
    Dictionary dictionary = new FileDictionary(new FileInputStream(path.toFile()));
    lookup.build(dictionary);

    List<String> inputs = NeuralNetworksUtils.generateInputs("music is");
    generateSuggestions(lookup, inputs);
  }

  @Test
  public void testFreeTextSuggesterFromIndex() throws Exception {
    Lookup lookup = new FreeTextSuggester(new WhitespaceAnalyzer());
    Directory directory = new RAMDirectory();

    IndexWriterConfig config = new IndexWriterConfig();
    IndexWriter writer = new IndexWriter(directory, config);

    for (String line : IOUtils.readLines(new FileInputStream("src/test/resources/billboard_lyrics_1964-2015.csv"))) {
      if (!line.startsWith("\"R")) {
        String[] fields = line.split(",");
        Document doc = new Document();
        doc.add(new TextField("lyrics", fields[4], Field.Store.YES));
        writer.addDocument(doc);
      }
    }
    writer.commit();
    writer.close();

    DirectoryReader reader = DirectoryReader.open(directory);
    Dictionary dictionary = new LuceneDictionary(reader, "lyrics");
    lookup.build(dictionary);

    List<String> inputs = NeuralNetworksUtils.generateInputs("music is my aircraft");
    generateSuggestions(lookup, inputs);
  }

  @Test
  public void testBuildNLMSuggesterFromModel() throws Exception {
    int miniBatchSize = 1000;
    int exampleLength = 1000;

    MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork("src/test/resources/models/ch4/charLSTM-130-55-3-2-0.1-XAVIER-RMSPROP-SOFTSIGN-253967-235740.zip");
    CharacterIterator charaterIterator = new CharacterIterator(getClass().getResource("/queries2.shuf.txt").getFile(),
            Charset.forName("ISO-8859-1"), miniBatchSize, exampleLength, CharacterIterator.getMinimalCharacterSet(), new Random());
    Lookup lookup = new CharLSTMNeuralLookup(network, charaterIterator);

    List<String> inputs = NeuralNetworksUtils.generateInputs("music is my aircraft");
    generateSuggestions(lookup, inputs);
  }

  @Test
  public void testBuildNLMSuggesterFromModelPlusWord2Vec() throws Exception {
    int miniBatchSize = 1000;
    int exampleLength = 1000;

    MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork("src/test/resources/models/ch4/charLSTM-130-55-3-2-0.1-XAVIER-RMSPROP-SOFTSIGN-253967-235740.zip");
    String file = getClass().getResource("/queries2.shuf.txt").getFile();
    CharacterIterator charaterIterator = new CharacterIterator(file,
            Charset.forName("ISO-8859-1"), miniBatchSize, exampleLength, CharacterIterator.getMinimalCharacterSet(), new Random());
    Word2Vec word2vec = new Word2Vec.Builder()
            .iterate(new FileDocumentIterator(file))
            .layerSize(100)
            .build();
    word2vec.fit();
    Lookup lookup = new CharLSTMWord2VecLookup(network, charaterIterator, word2vec);

    List<String> inputs = NeuralNetworksUtils.generateInputs("music is my aircraft");
    generateSuggestions(lookup, inputs);
  }

  @Test
  public void testBuildNLMSuggesterFromIndex() throws Exception {

    Directory directory = new RAMDirectory();
    try {
      String filePath = "src/test/resources/billboard_lyrics_1964-2015.csv";

      int lstmLayerSize = 50;
      int miniBatchSize = 1000;
      int exampleLength = 1000;
      int tbpttLength = 20;
      int numEpochs = 1;
      int noOfHiddenLayers = 1;
      double learningRate = 0.01;
      WeightInit weightInit = WeightInit.XAVIER;
      IUpdater updater = RmsProp.builder().learningRate(learningRate).build();
      Activation activation = Activation.TANH;

      Lookup lookup = new CharLSTMNeuralLookup(lstmLayerSize, miniBatchSize, exampleLength, tbpttLength, numEpochs,
              noOfHiddenLayers, weightInit, updater, activation);

      IndexWriterConfig config = new IndexWriterConfig();
      IndexWriter writer = new IndexWriter(directory, config);

      for (String line : IOUtils.readLines(new FileInputStream(filePath))) {
        if (!line.startsWith("\"R")) {
          String[] fields = line.split(",");
          Document doc = new Document();
          doc.add(new NumericDocValuesField("rank", Long.parseLong(fields[0])));
          doc.add(new TextField("lyrics", fields[4], Field.Store.YES));
          writer.addDocument(doc);
        }
      }
      writer.commit();

      DirectoryReader reader = DirectoryReader.open(directory);
      Dictionary dictionary = new DocumentDictionary(reader, "lyrics", "rank");
      lookup.build(dictionary);

      List<String> inputs = NeuralNetworksUtils.generateInputs("music is my aircraft");
      generateSuggestions(lookup, inputs);
    } finally {
      directory.close();
    }
  }

  @Test
  public void testBuildNLMSuggesterFromQueriesFile() throws Exception {

    int lstmLayerSize = 50;
    int miniBatchSize = 1000;
    int exampleLength = 1000;
    int tbpttLength = 20;
    int numEpochs = 1;
    int noOfHiddenLayers = 1;
    double learningRate = 0.01;
    WeightInit weightInit = WeightInit.XAVIER;
    IUpdater updater = RmsProp.builder().learningRate(learningRate).build();

    CharacterIterator iter = new CharacterIterator("src/test/resources/queries2.shuf.txt",
            Charset.forName("ISO-8859-1"), miniBatchSize, exampleLength, CharacterIterator.getMinimalCharacterSet(), new Random());

    MultiLayerNetwork net = NeuralNetworksUtils.trainLSTM(lstmLayerSize, tbpttLength, numEpochs, noOfHiddenLayers,
            iter, weightInit, updater, Activation.TANH, new ScoreIterationListener(100));

    Lookup lookup = new CharLSTMNeuralLookup(net, iter);
    List<String> inputs = NeuralNetworksUtils.generateInputs("music is my aircraft");
    generateSuggestions(lookup, inputs);

  }

  private void generateSuggestions(Lookup lookup, List<String> inputs) throws IOException {
    for (String input : inputs) {
      List<Lookup.LookupResult> lookupResults = lookup.lookup(input, false, 3);
      assertNotNull(lookupResults);
    }
  }

}
