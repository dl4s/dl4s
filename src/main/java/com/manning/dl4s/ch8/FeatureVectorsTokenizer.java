package com.manning.dl4s.ch8;

import org.apache.lucene.analysis.util.CharTokenizer;

/**
 * {@link CharTokenizer} which splits at whitespaces and commas
 */
class FeatureVectorsTokenizer extends CharTokenizer {
    @Override
    protected boolean isTokenChar(int c) {
      char c1 = Character.toChars(c)[0];
      return c1 != ',' && !Character.isWhitespace(c);
    }
  }