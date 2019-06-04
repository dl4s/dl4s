package com.manning.dl4s.ch7.encdec;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.xml.stream.XMLStreamException;

import com.manning.dl4s.ch7.ParallelSentence;
import com.manning.dl4s.ch7.TMXParser;
import com.manning.dl4s.utils.LuceneTokenizerFactory;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.LowerCaseFilterFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class to process a parallel corpus and generate related dictionaries and parallel sentences corpus.
 */
public class ParallelCorpusProcessor {

  private final Logger log = LoggerFactory.getLogger(getClass());

  private static final int MAX_DICT = 2000000;
  private static final String UNK = "<unk>";
  static final String EOS = "<eos>";
  static final String GO = "<go>";

  private final TokenizerFactory tokenizerFactory;

  private Set<String> dictSet = new HashSet<>();

  private Map<String, Double> freq = new HashMap<>();
  private Map<String, Double> dict = new HashMap<>();
  private Map<Double, String> revDict = new HashMap<>();

  /**
   * The contents of the corpus. This is a list of sentences (each word of the
   * sentence is denoted by a {@link Double}).
   */
  private final List<List<Double>> corpus = new ArrayList<>();

  private boolean countFreq;

  private File file;
  private final String sourceCode;
  private final String targetCode;
  private Collection<ParallelSentence> sentences;

  public ParallelCorpusProcessor(String filename, String sourceCode, String targetCode, boolean countFreq) throws IOException {
    this(new File(filename), sourceCode, targetCode, countFreq);
  }

  public ParallelCorpusProcessor(File file, String sourceCode, String targetCode, boolean countFreq) throws IOException {
    this.file = file;
    this.sourceCode = sourceCode;
    this.targetCode = targetCode;
    this.countFreq = countFreq;
    Analyzer simpleAnalyzer = CustomAnalyzer.builder()
        .withTokenizer(StandardTokenizerFactory.class)
        .addTokenFilter(LowerCaseFilterFactory.class)
        .build();
    tokenizerFactory = new LuceneTokenizerFactory(simpleAnalyzer);
  }

  public void process() throws IOException, XMLStreamException {

    dict.put(UNK, 0.0);
    revDict.put(0.0, UNK);
    dict.put(EOS, 1.0);
    revDict.put(1.0, EOS);
    dict.put(GO, 2.0);
    revDict.put(2.0, GO);

    log.info("Building the parallel dictionary...");
    TMXParser tmxParser = new TMXParser(file, sourceCode, targetCode);
    sentences = tmxParser.parse();
    for (ParallelSentence sentence : sentences) {
      processLine(sentence.getSource());
      processLine(sentence.getTarget());
    }
    Map<String, Double> freqs = getFreq();
    Map<Double, Set<String>> freqMap = new TreeMap<>((o1, o2) -> (int) (o2 - o1)); // tokens of the same frequency fall under the same key, the order is reversed so the most frequent tokens go first
    for (Map.Entry<String, Double> entry : freqs.entrySet()) {
      Set<String> set = freqMap.computeIfAbsent(entry.getValue(), k -> new TreeSet<>());
      // tokens of the same frequency would be sorted alphabetically
      set.add(entry.getKey());
    }
    int cnt = 0;
    // the tokens order is preserved for TreeSet
    Set<String> dictSet = new TreeSet<>(dict.keySet());
    // get most frequent tokens and put them to dictSet
    for (Map.Entry<Double, Set<String>> entry : freqMap.entrySet()) {
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

    double idx = 3.0;
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

    for (ParallelSentence sentence : sentences) {
      String source = sentence.getSource();
      List<String> words = tokenizerFactory.create(source).getTokens();
      corpus.add(wordsToIndexes(words));
      String target = sentence.getTarget();
      words = tokenizerFactory.create(target).getTokens();
      corpus.add(wordsToIndexes(words));
    }

    log.info("Done. Corpus size is {}", corpus.size());
  }

  protected void processLine(String lastLine) {
    tokenizeLine(lastLine, dictSet);
  }

  protected void tokenizeLine(String lastLine, Collection<String> dictSet) {

    List<String> tokens = tokenizerFactory.create(lastLine).getTokens();

    for (String word : tokens) {
      addWord(dictSet, word);
    }
  }

  private void addWord(Collection<String> coll, String word) {
    if (coll != null) {
      coll.add(word);
    }
    if (countFreq) {
      Double count = freq.get(word);
      if (count == null) {
        freq.put(word, 1.0);
      } else {
        freq.put(word, count + 1);
      }
    }
  }

  public Map<String, Double> getFreq() {
    return freq;
  }

  /**
   * Converts an iterable sequence of words to a list of indices. This will
   * never return {@code null} but may return an empty {@link List}.
   *
   * @param words
   *            sequence of words
   * @return list of indices.
   */
  protected final List<Double> wordsToIndexes(final Iterable<String> words) {
    int i = MTNetwork.ROW_SIZE;
    final List<Double> wordIdxs = new LinkedList<>();
    for (final String word : words) {
      if (--i == 0) {
        break;
      }
      final Double wordIdx = dict.get(word);
      if (wordIdx != null) {
        wordIdxs.add(wordIdx);
      } else {
        wordIdxs.add(0.0);
      }
    }
    return wordIdxs;
  }

  public List<String> indexesToWords(List<Double> row) {
    List<String> sentence = new ArrayList<>(row.size());

    for (Double i : row) {
      sentence.add(revDict.get(i));
    }

    return sentence;
  }

  public void setRevDict(Map<Double, String> revDict) {
    this.revDict = revDict;
  }

  public Map<String, Double> getDict() {
    return dict;
  }

  public List<List<Double>> getCorpus() {
    return corpus;
  }

  public Map<Double, String> getRevDict() {
    return revDict;
  }

  public Collection<ParallelSentence> getSentences() {
    return sentences;
  }
}
