package com.manning.dl4s.ch2;

import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.junit.Test;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Trivial tests for indexing and search in Lucene
 */
public class SamplesTest {

  @Test
  public void testIndexTwoFieldsSearchAndFeedWord2Vec() throws Exception {
    Path path = Paths.get("target/idx");
    Directory directory = FSDirectory.open(path);

    try {

      Analyzer defaultAnalyzer = new EnglishAnalyzer();
      Map<String, Analyzer> perFieldAnalyzers = new HashMap<>();

      CharArraySet stopWords = new CharArraySet(Arrays.asList("a", "an", "the"), true);
      perFieldAnalyzers.put("pages", new StopAnalyzer(stopWords));
      perFieldAnalyzers.put("title", new WhitespaceAnalyzer());

      Analyzer analyzer = new PerFieldAnalyzerWrapper(defaultAnalyzer, perFieldAnalyzers);
      IndexWriterConfig config = new IndexWriterConfig(analyzer);
      IndexWriter writer = new IndexWriter(directory, config);

      Document doc1 = new Document();
      doc1.add(new TextField("title", "Deep learning for search", Field.Store.YES));
      doc1.add(new TextField("page", "Living in the information age ...", Field.Store.YES));

      Document doc2 = new Document();
      doc2.add(new TextField("title", "Relevant search", Field.Store.YES));
      doc2.add(new TextField("page", "Getting a search engine to behave can be maddening ...", Field.Store.YES));

      writer.addDocument(doc1);
      writer.addDocument(doc2);

      writer.commit();
      writer.close();

      IndexReader reader = DirectoryReader.open(directory);
      try {
        IndexSearcher searcher = new IndexSearcher(reader);
        QueryParser parser = new QueryParser("title", new WhitespaceAnalyzer());
        Query query = parser.parse("+search");
        TopDocs hits = searcher.search(query, 10);
        for (int i = 0; i < hits.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = hits.scoreDocs[i];
          Explanation explanation = searcher.explain(query, scoreDoc.doc);
          System.out.println(explanation);
          Document doc = searcher.doc(scoreDoc.doc);
          System.out.println("--");
          System.out.println(doc.get("title") + " : " + scoreDoc.score);
        }

        FieldValuesSentenceIterator iterator = new FieldValuesSentenceIterator(reader,"page");

        Word2Vec vec = new Word2Vec.Builder()
            .layerSize(100)
            .windowSize(2)
            .tokenizerFactory(new DefaultTokenizerFactory())
            .iterate(iterator)
            .build();

        vec.fit();

        Collection<String> search = vec.wordsNearestSum("search", 2);
        System.out.println(search);

      } finally {
        reader.close();
        directory.close();
      }

    } finally {
      directory.close();
    }
  }

  @Test
  public void testAnalyzeWithWSAndSWs() throws Exception {
    CharArraySet stopWords = new CharArraySet(Arrays.asList("a", "an", "the"), true);
    Analyzer a = new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer t = new WhitespaceTokenizer();
        return new TokenStreamComponents(t, new StopFilter(t, stopWords));
      }
    };
    TokenStream ts = a.tokenStream(null, "I like search engines");
    StringWriter w = new StringWriter();
    new TokenStreamToDot(null, ts, new PrintWriter(w)).toDot();
    System.out.println(w);
  }

}