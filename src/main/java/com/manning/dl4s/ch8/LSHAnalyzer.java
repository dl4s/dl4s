package com.manning.dl4s.ch8;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.minhash.MinHashFilter;
import org.apache.lucene.analysis.shingle.ShingleFilter;

/**
 * {@link Analyzer} for LSH search
 */
public class LSHAnalyzer extends Analyzer {

  private static final int DEFAULT_SHINGLE_SIZE = 5;

  private final int min;
  private final int max;
  private final int hashCount;
  private final int bucketCount;
  private final int hashSetSize;

  private LSHAnalyzer(int min, int max, int hashCount, int bucketCount, int hashSetSize) {
    super();
    this.min = min;
    this.max = max;
    this.hashCount = hashCount;
    this.bucketCount = bucketCount;
    this.hashSetSize = hashSetSize;
  }

  public LSHAnalyzer() {
    this(DEFAULT_SHINGLE_SIZE, DEFAULT_SHINGLE_SIZE, MinHashFilter.DEFAULT_HASH_COUNT, MinHashFilter.DEFAULT_BUCKET_COUNT, MinHashFilter.DEFAULT_HASH_SET_SIZE);
  }

  @Override
  protected TokenStreamComponents createComponents(String fieldName) {
    Tokenizer source = new FeatureVectorsTokenizer();
    TokenFilter truncate = new TruncateTokenFilter(source, 3);
    TokenFilter featurePos = new FeaturePositionTokenFilter(truncate);
    ShingleFilter shingleFilter = new ShingleFilter(featurePos, min, max);
    shingleFilter.setTokenSeparator(" ");
    shingleFilter.setOutputUnigrams(false);
    shingleFilter.setOutputUnigramsIfNoShingles(false);
    TokenStream filter = new MinHashFilter(shingleFilter, hashCount, bucketCount, hashSetSize, bucketCount > 1);
    return new TokenStreamComponents(source, filter);
  }

}
