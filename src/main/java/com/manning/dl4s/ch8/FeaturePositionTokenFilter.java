package com.manning.dl4s.ch8;

import java.io.IOException;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

/**
 * {@link TokenFilter} which prepends the token / feature position plus underscore to the token itself
 */
final class FeaturePositionTokenFilter extends TokenFilter {

    private final CharTermAttribute termAttribute = addAttribute(CharTermAttribute.class);
    private int tokenCount = 0;

    FeaturePositionTokenFilter(TokenStream stream) {
      super(stream);
    }

    @Override
    public boolean incrementToken() throws IOException {
      if (input.incrementToken()) {
        tokenCount++;
        String token = new String(termAttribute.buffer(), 0, termAttribute.length());
        termAttribute.setEmpty();
        termAttribute.append(String.valueOf(tokenCount));
        termAttribute.append("_");
        termAttribute.append(token);
        return true;
      } else {
        return false;
      }
    }

  @Override
  public void reset() throws IOException {
    super.reset();
    tokenCount = 0;
  }

}