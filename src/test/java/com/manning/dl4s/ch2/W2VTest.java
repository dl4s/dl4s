package com.manning.dl4s.ch2;

import com.manning.dl4s.utils.LuceneTokenizerFactory;
import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.LinkedList;

import static org.junit.Assert.assertFalse;

/**
 * Tests for {@link Word2Vec} based synonym expansion
 */
public class W2VTest {

  private static Word2Vec vec;

  @BeforeClass
  public static void setUpBefore() throws Exception {
    String filePath = "src/test/resources/billboard_lyrics_1964-2015.csv";
    if (!new File(filePath).exists()) {
      URI uri = URI.create("https://github.com/walkerkq/musiclyrics/raw/master/billboard_lyrics_1964-2015.csv");
      InputStream in = uri.toURL().openStream();
      Files.copy(in, Paths.get(filePath));
      in.close();
    }
    // note that in practice using a csv as plain text input is not a good idea
    SentenceIterator iter = new BasicLineIterator(filePath);

    vec = new Word2Vec.Builder()
        .layerSize(60)
        .windowSize(3)
        .epochs(5)
        .elementsLearningAlgorithm(new SkipGram<>())
        .tokenizerFactory(new LuceneTokenizerFactory(new StandardAnalyzer()))
        .iterate(iter)
        .build();

    vec.fit();
  }

  @Test
  public void testSynonymsFiltering() {
    String[] testWords = new String[]{"love", "eat", "like", "hate", "nice", "cool", "bad", "sad"};
    for (String tw : testWords) {
      Collection<String> wordsNearest = vec.wordsNearest(tw, 3);
      Collection<String> all = new LinkedList<>();
      for (String wn : wordsNearest) {
        double similarity = vec.similarity(tw, wn);
        if (similarity > 0.3) {
          all.add(wn);
        }

      }
      wordsNearest.retainAll(all);
      assertFalse(wordsNearest.isEmpty());
    }
  }

  @Test
  public void testW2VSimWithBillboard() {
    String[] words = new String[] {"guitar", "love", "party"};

    int n = 3;
    for (String word : words) {
      Collection<String> lst = vec.wordsNearestSum(word, n);
      assertFalse(lst.isEmpty());
    }
  }

  @Test
  public void testAnalyzeWithW2VSynonyms() throws Exception {

    Analyzer a = new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer tokenizer = new WhitespaceTokenizer();
        double minAcc = 0.95;
        TokenFilter synFilter = new W2VSynonymFilter(tokenizer, vec, minAcc);
        return new TokenStreamComponents(tokenizer, synFilter);
      }
    };
    TokenStream ts = a.tokenStream(null, "I like joy spiked with pain and music is my airplane");
    StringWriter w = new StringWriter();
    new TokenStreamToDot(null, ts, new PrintWriter(w)).toDot();
    System.out.println(w);
  }
}
