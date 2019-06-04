package com.manning.dl4s.ch7;

/**
 * A sentence having both texts in the source language and in the target language.
 */
public class ParallelSentence {

  private final String source;
  private final String target;

  public ParallelSentence(String source, String target) {
    this.source = source;
    this.target = target;
  }

  public String getSource() {
    return source;
  }

  public String getTarget() {
    return target;
  }
}