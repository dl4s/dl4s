package com.manning.dl4s.ch9.functions;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.manning.dl4s.ch5.FieldValuesLabelAwareIterator;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

/**
 * Utility methods for the tweet ingest and learn application
 */
public class Utils {

  public static Word2Vec fetchWordVectors(Path indexPath) throws IOException {
    Directory dir = FSDirectory.open(indexPath);
    Word2Vec word2Vec;
    if (dir.listAll() != null && dir.listAll().length > 0) {
      DirectoryReader reader = DirectoryReader.open(dir);

      DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
      word2Vec = new Word2Vec.Builder()
              .tokenizerFactory(tokenizerFactory)
              .layerSize(60)
              .epochs(1)
              .useUnknown(true)
              .iterate(new FieldValuesLabelAwareIterator(reader, "text"))
              .build();
      word2Vec.fit();

      reader.close();
    } else {
      String path = "src/test/resources/sentences.txt";
      DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
      word2Vec = new Word2Vec.Builder()
              .tokenizerFactory(tokenizerFactory)
              .layerSize(60)
              .epochs(1)
              .useUnknown(true)
              .iterate(new FileDocumentIterator(path))
              .build();
      word2Vec.fit();
    }
    dir.close();
    return word2Vec;
  }

  public static ParagraphVectors fetchParagraghVectors(Path indexPath) throws IOException {
    Directory dir = FSDirectory.open(indexPath);
    ParagraphVectors paragraphVectors;
    if (dir.listAll() != null && dir.listAll().length > 0) {
      DirectoryReader reader = DirectoryReader.open(dir);

      DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
      paragraphVectors = new ParagraphVectors.Builder()
          .tokenizerFactory(tokenizerFactory)
          .trainWordVectors(true)
          .layerSize(60)
          .epochs(1)
          .useUnknown(true)
          .iterate(new FieldValuesLabelAwareIterator(reader, "text"))
          .build();
      paragraphVectors.fit();

      reader.close();
    } else {
      String path = "src/test/resources/sentences.txt";
      DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
      paragraphVectors = new ParagraphVectors.Builder()
          .tokenizerFactory(tokenizerFactory)
          .trainWordVectors(true)
          .layerSize(60)
          .epochs(1)
          .useUnknown(true)
          .iterate(new FileDocumentIterator(path))
          .build();
      paragraphVectors.fit();
    }
    dir.close();
    return paragraphVectors;
  }

  public static ParagraphVectors fetchParagraphVectors() throws IOException {
    return fetchParagraghVectors(getIndexPath());
  }

  public static Word2Vec fetchWordVectors() throws IOException {
    return fetchWordVectors(getIndexPath());
  }

  public static Path getIndexPath() {
    return Paths.get("src/test/resources/stream_index");
  }
}
