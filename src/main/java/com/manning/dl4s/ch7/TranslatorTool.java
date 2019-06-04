package com.manning.dl4s.ch7;

import java.util.Collection;

/**
 * Simple machine translation (prediction) API
 */
public interface TranslatorTool {

  Collection<Translation> translate(String text);

  class Translation {

    private final String translationString;
    private final Double score;

    public Translation(String translationString, Double score) {
      this.translationString = translationString;
      this.score = score;
    }

    public String getTranslationString() {
      return translationString;
    }

    public Double getScore() {
      return score;
    }

    @Override
    public String toString() {
      return "Translation{" +
              "translationString='" + translationString + '\'' +
              ", score=" + score +
              '}';
    }
  }
}