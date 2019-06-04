package com.manning.dl4s.ch8;

import java.io.IOException;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;

/**
 * {@link TokenFilter} which truncates a token bigger than {#length}.
 */
class TruncateTokenFilter extends TokenFilter {

    private final CharTermAttribute termAttribute = addAttribute(CharTermAttribute.class);
    private final KeywordAttribute keywordAttr = addAttribute(KeywordAttribute.class);

    private final int length;

    TruncateTokenFilter(TokenStream input, int length) {
      super(input);
      if (length < 1) {
        throw new IllegalArgumentException("length parameter must be a positive number: " + length);
      }
      this.length = length;
    }

    @Override
    public final boolean incrementToken() throws IOException {
      if (input.incrementToken()) {
        if (!keywordAttr.isKeyword() && termAttribute.length() > length) {
          termAttribute.setLength(length);
        }
        return true;
      } else {
        return false;
      }
    }
  }