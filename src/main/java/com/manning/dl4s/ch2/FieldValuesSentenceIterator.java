package com.manning.dl4s.ch2;

import java.io.IOException;
import java.util.Collections;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

/**
 * {@link SequenceIterator} over field values of Lucene documents
 */
public class FieldValuesSentenceIterator implements SentenceIterator {

  private final IndexReader reader;
  private final String field;
  private int currentId;
  private SentencePreProcessor preProcessor;

  public FieldValuesSentenceIterator(IndexReader reader, String field) {
    this.reader = reader;
    this.field = field;
    this.currentId = 0;
  }

  @Override
  public String nextSentence() {
    if (!hasNext()) {
      return null;
    }
    try {
      Document document = reader.document(currentId, Collections.singleton(field));
      String sentence = document.getField(field).stringValue();
      return preProcessor != null ? preProcessor.preProcess(sentence) : sentence;
    } catch (IOException e) {
      throw new RuntimeException(e);
    } finally {
      currentId++;
    }
  }

  @Override
  public boolean hasNext() {
    return currentId < reader.numDocs();
  }

  @Override
  public void reset() {
    currentId = 0;
  }

  @Override
  public void finish() {
  }

  @Override
  public SentencePreProcessor getPreProcessor() {
    return this.preProcessor;
  }

  @Override
  public void setPreProcessor(SentencePreProcessor preProcessor) {
    this.preProcessor = preProcessor;
  }
}
