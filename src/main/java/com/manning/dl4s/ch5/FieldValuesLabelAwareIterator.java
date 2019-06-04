package com.manning.dl4s.ch5;

import java.io.IOException;
import java.util.Collections;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;

/**
 * Iterate over documents fetched from the index, labels are composed by "doc_" prefix plus each document number.
 */
public class FieldValuesLabelAwareIterator implements LabelAwareIterator {

  private final IndexReader reader;
  private final String field;
  private int currentId;

  public FieldValuesLabelAwareIterator(IndexReader reader, String field) {
    this.reader = reader;
    this.field = field;
    this.currentId = 0;
  }

  @Override
  public boolean hasNextDocument() {
    return currentId < reader.maxDoc();
  }

  @Override
  public LabelledDocument nextDocument() {
    if (!hasNext()) {
      return null;
    }
    try {
      LabelledDocument labelledDocument = new LabelledDocument();
      Document document = reader.document(currentId, Collections.singleton(field));
      labelledDocument.addLabel("doc_" + currentId);
      labelledDocument.setId("doc_" + currentId);
      labelledDocument.setContent(document.getField(field).stringValue());
      return labelledDocument;
    } catch (IOException e) {
      throw new RuntimeException(e);
    } finally {
      currentId++;
    }
  }

  @Override
  public void reset() {
    currentId = 0;
  }

  @Override
  public LabelsSource getLabelsSource() {
    return new LabelsSource("doc_" + currentId);
  }

  @Override
  public void shutdown() {
  }

  @Override
  public boolean hasNext() {
    return hasNextDocument();
  }

  @Override
  public LabelledDocument next() {
    return nextDocument();
  }
}
