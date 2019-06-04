package com.manning.dl4s.ch6;

import com.google.common.io.Files;
import com.manning.dl4s.ch5.FieldValuesLabelAwareIterator;
import com.manning.dl4s.utils.WikipediaImport;
import com.manning.dl4s.utils.encdec.EncoderDecoderLSTM;
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.classification.ClassificationResult;
import org.apache.lucene.classification.Classifier;
import org.apache.lucene.classification.KNearestNeighborClassifier;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.queries.mlt.MoreLikeThis;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.StringReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static org.junit.Assert.assertNotNull;

/**
 * Tests for chapter 6
 */
public class RelatedContentTest {

  @Test
  public void testRelatedContent() throws Exception {
    Directory directory = new RAMDirectory();

    try {

      IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());

      IndexWriter writer = new IndexWriter(directory, config);
      FieldType ft = new FieldType(TextField.TYPE_STORED);
      ft.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
      ft.setTokenized(true);
      ft.setStored(true);
      ft.setStoreTermVectors(true);
      String fieldName = "text";

      String pathname = "src/test/resources/enwiki-20180820-pages-articles14.xml-p7697599p7744799";
      String languageCode = "en";

      File dump = new File(pathname);
      WikipediaImport wikipediaImport = new WikipediaImport(dump, languageCode, true);
      wikipediaImport.importWikipedia(writer, ft);

      IndexReader reader = DirectoryReader.open(writer);

      FieldValuesLabelAwareIterator iterator = new FieldValuesLabelAwareIterator(reader, fieldName);

      DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
      tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

      int epochs = 2;
      int layerSize = 200;
      int windowSize = 3;
      int seed = 12345;

      Word2Vec vec = new Word2Vec.Builder()
          .layerSize(layerSize)
          .windowSize(windowSize)
          .tokenizerFactory(tokenizerFactory)
          .iterate(iterator)
          .epochs(epochs)
          .seed(seed)
          .build();
      vec.fit();

      Random r = new Random();
      Collection<String> queries = new LinkedList<>();
      queries.add("Ledgewood Circle");
      VocabCache<VocabWord> vocabCache = vec.getLookupTable().getVocabCache();
      for (int i = 0; i < 100; i++) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int j = 0; j < 1 + r.nextInt(4); j++) {
          if (stringBuilder.length() > 0) {
            stringBuilder.append(' ');
          }
          String word = null;
          try {
            word = vocabCache.wordAtIndex(r.nextInt(vocabCache.numWords()));
          } catch (Exception e) {
            // ignore
          }
          if (word != null) {
            stringBuilder.append(word);
          }
        }
        String e = stringBuilder.toString();
        if (e.trim().length() > 0) {
          queries.add(e);
        }
      }

      ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
          .iterate(iterator)
          .layerSize(layerSize)
          .seed(seed)
          .epochs(epochs)
          .tokenizerFactory(tokenizerFactory)
          .useExistingWordVectors(vec)
          .build();
      paragraphVectors.fit();

      Similarity[] similarities = new Similarity[] {new BM25Similarity(), new ClassicSimilarity(), new LMDirichletSimilarity()};

      Analyzer analyzer = new WhitespaceAnalyzer();
      MoreLikeThis moreLikeThis = new MoreLikeThis(reader);
      moreLikeThis.setAnalyzer(analyzer);

      try {
        IndexSearcher searcher = new IndexSearcher(reader);
        int topN = 10;

        QueryParser parser = new QueryParser(fieldName, analyzer);

        double avgAcc = getAvgAcc(reader, queries, paragraphVectors, searcher, topN, parser);
        System.out.println("pv avgAcc " + avgAcc);

        for (Similarity similarity : similarities) {
          searcher.setSimilarity(similarity);

          double avgAccSim = getAvgAcc(fieldName, reader, queries, moreLikeThis, searcher, topN, parser);
          System.out.println(similarity + " avgAcc = " + avgAccSim);
        }
      } finally {
        writer.close();
        reader.close();
      }

    } finally {
      directory.close();
    }
  }

  private double getAvgAcc(String fieldName, IndexReader reader, Collection<String> queries, MoreLikeThis moreLikeThis, IndexSearcher searcher, int topN, QueryParser parser) {
    double avgAccSim = 0;
    for (String queryString : queries) {
      try {
        Query query = parser.parse(StringEscapeUtils.escapeJava(queryString));
        TopDocs hits = searcher.search(query, 10);
        for (int i = 0; i < hits.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = hits.scoreDocs[i];
          Document doc = searcher.doc(scoreDoc.doc);

          String s = doc.get(fieldName);
          Query simQuery = moreLikeThis.like(fieldName, new StringReader(s));

          String[] originalCategories = doc.getValues(WikipediaImport.CATEGORY);
          Arrays.sort(originalCategories);
          try {
            TopDocs related = searcher.search(simQuery, topN);
            double acc = 0;
            for (ScoreDoc rd : related.scoreDocs) {
              if (rd.doc == scoreDoc.doc) {
                continue;
              }
              Document document = reader.document(rd.doc);

              String[] categories = document.getValues(WikipediaImport.CATEGORY);
              if (categories != null) {
                Arrays.sort(categories);
                for (String c : categories) {
                  if (Arrays.binarySearch(originalCategories, c) >= 0) {
                    acc += 1d;
                    break;
                  }
                }
              }
            }
            acc /= topN;
            avgAccSim += acc;
          } catch (Exception e) {
            // do nothing
          }

        }


      } catch (Throwable e) {
        // do nothing
      }
    }

    avgAccSim /= (double) queries.size();
    return avgAccSim;
  }

  private double getAvgAcc(IndexReader reader, Collection<String> queries, ParagraphVectors paragraphVectors, IndexSearcher searcher, int topN, QueryParser parser) {
    double avgAcc = 0;
    for (String queryString : queries) {
      try {

        Query query = parser.parse(StringEscapeUtils.escapeJava(queryString));
        TopDocs hits = searcher.search(query, 10);
        for (int i = 0; i < hits.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = hits.scoreDocs[i];
          Document doc = searcher.doc(scoreDoc.doc);

          String title = doc.get(WikipediaImport.TITLE);
          String[] originalCategories = doc.getValues(WikipediaImport.CATEGORY);
          Arrays.sort(originalCategories);

          String label = "doc_" + scoreDoc.doc;
          try {
            INDArray labelVector = paragraphVectors.getLookupTable().vector(label);
            if (labelVector == null) {
              LabelledDocument document = new LabelledDocument();
              document.setLabels(Collections.singletonList(label));
              document.setContent(title);
              labelVector = paragraphVectors.inferVector(document);
            }
            Collection<String> docIds = paragraphVectors.nearestLabels(labelVector, topN);
            double acc = 0;
            for (String docId : docIds) {
              Document document = reader.document(Integer.parseInt(docId.substring(4)));
              String[] categories = document.getValues(WikipediaImport.CATEGORY);
              if (categories != null) {
                Arrays.sort(categories);
                Arrays.sort(categories);
                for (String c : categories) {
                  if (Arrays.binarySearch(originalCategories, c) >= 0) {
                    acc += 1d;
                    break;
                  }
                }
              }
            }
            acc /= topN;
            avgAcc += acc;
          } catch (Exception e) {
            // do nothing
          }
        }
      } catch (Exception e) {
        // do nothing
      }
    }
    avgAcc /= (double) queries.size();
    return avgAcc;
  }


  @Test
  public void testThoughtVectorsExtraction() throws Exception {
    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);

    try {

      IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());

      IndexWriter writer = new IndexWriter(directory, config);
      FieldType ft = new FieldType(TextField.TYPE_STORED);
      ft.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
      ft.setTokenized(true);
      ft.setStored(true);
      String fieldName = "text";

      String pathname = "src/test/resources/enwiki-20180820-pages-articles14.xml-p7697599p7744799";
      String languageCode = "en";

      File dump = new File(pathname);
      WikipediaImport wikipediaImport = new WikipediaImport(dump, languageCode, true);
      wikipediaImport.importWikipedia(writer, ft);

      IndexReader reader = DirectoryReader.open(writer);

      File modelFile = new File("target/tv-encdec-" + System.currentTimeMillis());
      File corpusFile = new File(pathname);

      EncoderDecoderLSTM encoderDecoderLSTM = new EncoderDecoderLSTM(modelFile, corpusFile);

      try {
        IndexSearcher searcher = new IndexSearcher(reader);

        QueryParser parser = new QueryParser(fieldName, new WhitespaceAnalyzer());
        Query query = parser.parse("travel hints for south america");
        TopDocs hits = searcher.search(query, 10);
        for (int i = 0; i < hits.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = hits.scoreDocs[i];
          Document doc = searcher.doc(scoreDoc.doc);

          String[] tokens = doc.get(fieldName).split(" ");
          INDArray thoughtVector = encoderDecoderLSTM.getThoughtVector(tokens);
          assertNotNull(thoughtVector);
        }

      } finally {
        writer.deleteAll();
        writer.commit();
        writer.close();
        reader.close();
      }

    } finally {
      directory.close();
    }
  }

  @Test
  public void testRelatedContentWithClassifiers() throws Exception {
    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);

    try {

      IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());

      IndexWriter writer = new IndexWriter(directory, config);
      FieldType ft = new FieldType(TextField.TYPE_STORED);
      ft.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
      ft.setTokenized(true);
      ft.setStored(true);
      String fieldName = WikipediaImport.TEXT;

      String pathname = "src/test/resources/enwiki-20180820-pages-articles14.xml-p7697599p7744799";
      String languageCode = "en";

      File dump = new File(pathname);
      WikipediaImport wikipediaImport = new WikipediaImport(dump, languageCode, true);
      wikipediaImport.importWikipedia(writer, ft);

      IndexReader reader = DirectoryReader.open(writer);

      MoreLikeThis moreLikeThis = new MoreLikeThis(reader);
      moreLikeThis.setAnalyzer(writer.getAnalyzer());

      String classFieldName = WikipediaImport.CATEGORY;

      Classifier<BytesRef> classifier = new KNearestNeighborClassifier(reader, new BM25Similarity(), writer.getAnalyzer(), null, 3, 1, 1, classFieldName, fieldName);
      try {
        IndexSearcher searcher = new IndexSearcher(reader);

        QueryParser parser = new QueryParser(fieldName, writer.getAnalyzer());
        QueryParser classParser = new QueryParser(classFieldName, writer.getAnalyzer());
        Query query = parser.parse("travel hints for south america");
        TopDocs hits = searcher.search(query, 10);
        for (int i = 0; i < hits.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = hits.scoreDocs[i];
          Document doc = searcher.doc(scoreDoc.doc);

          BooleanQuery.Builder actualCategoriesQuery = new BooleanQuery.Builder();
          String[] actualCategories = doc.getValues(classFieldName);
          for (String actualCategory : actualCategories) {
            actualCategoriesQuery.add(new BooleanClause(classParser.parse(actualCategory), BooleanClause.Occur.SHOULD));
          }

          String s = doc.get(fieldName);
          Query simQuery = moreLikeThis.like(fieldName, new StringReader(s));

          BooleanQuery relatedContentQuery = new BooleanQuery.Builder()
              .add(actualCategoriesQuery.build(), BooleanClause.Occur.MUST)
              .add(simQuery, BooleanClause.Occur.SHOULD)
              .build();
          int topN = 5;
          TopDocs related = searcher.search(relatedContentQuery, topN);
          for (ScoreDoc rd : related.scoreDocs) {
            if (rd.doc == scoreDoc.doc) {
              continue;
            }
            Document document = reader.document(rd.doc);
            System.out.println("mlt + cat " + " -> " + document.get(WikipediaImport.TITLE) + " (" + Arrays.toString(actualCategories) + ")");
          }

          String text = doc.get(fieldName);
          ClassificationResult<BytesRef> classificationResult = classifier.assignClass(text);

          BytesRef classifiedCategory = classificationResult.getAssignedClass();

          relatedContentQuery = new BooleanQuery.Builder()
              .add(new BooleanClause(classParser.parse(classifiedCategory.utf8ToString()), BooleanClause.Occur.MUST))
              .add(simQuery, BooleanClause.Occur.SHOULD)
              .build();
          related = searcher.search(relatedContentQuery, topN);
          for (ScoreDoc rd : related.scoreDocs) {
            if (rd.doc == scoreDoc.doc) {
              continue;
            }
            Document document = reader.document(rd.doc);
            System.out.println("mlt + classifier -> " + document.get(WikipediaImport.TITLE) + " (" + classifiedCategory.utf8ToString() + ")");
          }

        }
      } finally {
        writer.deleteAll();
        writer.commit();
        writer.close();
        reader.close();
      }
    } finally {
      directory.close();
    }
  }

}
