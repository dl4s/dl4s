package com.manning.dl4s.ch5;

import com.google.common.io.Files;
import com.manning.dl4s.ch2.FieldValuesSentenceIterator;
import com.manning.dl4s.utils.VectorizeUtils;
import com.manning.dl4s.utils.WikipediaImport;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Ranking tests evaluating different {@link Similarity} implementations against {@link WordEmbeddingsSimilarity}
 */
public class RankingTest {

  @Test
  public void testRankingWithDifferentSimilarities() throws Exception {
    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);

    try {

      Collection<Similarity> similarities = Arrays.asList(new ClassicSimilarity(), new BM25Similarity(2.5f, 0.2f),
          new LMJelinekMercerSimilarity(0.1f), new MultiSimilarity(new Similarity[] {new BM25Similarity(), new LMJelinekMercerSimilarity(0.1f)}));
      for (Similarity similarity : similarities) {

        IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());
        config.setSimilarity(similarity);

        IndexWriter writer = new IndexWriter(directory, config);
        FieldType ft = new FieldType(TextField.TYPE_STORED);
        ft.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
        ft.setTokenized(true);
        ft.setStored(true);
        ft.setStoreTermVectors(true);
        ft.setStoreTermVectorOffsets(true);
        ft.setStoreTermVectorPositions(true);

        Document doc1 = new Document();
        doc1.add(new Field("title", "riemann bernhard - life and works of bernhard riemann", ft));

        Document doc2 = new Document();
        doc2.add(new Field("title", "thomas bernhard biography - bio and influence in literature", ft));

        Document doc3 = new Document();
        doc3.add(new Field("title", "riemann hypothesis - a deep dive into a mathematical mystery", ft));

        Document doc4 = new Document();
        doc4.add(new Field("title", "bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard", ft));

        writer.addDocument(doc1);
        writer.addDocument(doc2);
        writer.addDocument(doc3);
        writer.addDocument(doc4);

        writer.commit();

        IndexReader reader = DirectoryReader.open(directory);
        try {
          IndexSearcher searcher = new IndexSearcher(reader);
          Terms fieldTerms = MultiFields.getTerms(reader, "title");

          System.out.println(similarity);
          searcher.setSimilarity(similarity);
          QueryParser parser = new QueryParser("title", new WhitespaceAnalyzer());
          String queryString = "bernhard riemann influence";
          Query query = parser.parse(queryString);
          TopDocs hits = searcher.search(query, 10);
          for (int i = 0; i < hits.scoreDocs.length; i++) {
            ScoreDoc scoreDoc = hits.scoreDocs[i];
            Document doc = searcher.doc(scoreDoc.doc);

            String title = doc.get("title");
            System.out.println(title + " : " + scoreDoc.score);
            if (similarity instanceof ClassicSimilarity) {
              double[] queryVector = new double[(int) fieldTerms.size()];
              Arrays.fill(queryVector, 0d);
              for (String queryTerm : queryString.split(" ")) {
                TermsEnum iterator = fieldTerms.iterator();
                int j = 0;
                BytesRef term;
                while ((term = iterator.next()) != null) {

                  TermsEnum.SeekStatus seekStatus = iterator.seekCeil(term);
                  if (seekStatus.equals(TermsEnum.SeekStatus.END)) {
                    iterator = fieldTerms.iterator();
                  }
                  if (seekStatus.equals(TermsEnum.SeekStatus.FOUND)) {
                    if (term.utf8ToString().equals(queryTerm)) {
                      double tf = iterator.totalTermFreq();
                      double docFreq = iterator.docFreq();
                      queryVector[j] = VectorizeUtils.tfIdf(reader.numDocs(), tf, docFreq);
                    }
                  }
                  j++;
                }
              }

              Terms docTerms = reader.getTermVector(scoreDoc.doc, "title");
              double[] documentVector = VectorizeUtils.toSparseTFIDFDoubleArray(docTerms, fieldTerms, reader.numDocs());

              System.out.println("cosineSimilarity=" + VectorizeUtils.cosineSimilarity(queryVector, documentVector));
            }
          }

        } finally {
          writer.commit();
          writer.close();
          reader.close();
        }
      }

    } finally {
      directory.close();
    }
  }

  @Test
  public void testRankingWithTFIDFAveragedWordEmbeddings() throws Exception {
    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);

    try {

      IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());

      IndexWriter writer = new IndexWriter(directory, config);
      FieldType ft = new FieldType(TextField.TYPE_STORED);
      ft.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
      ft.setTokenized(true);
      ft.setStored(true);
      ft.setStoreTermVectors(true);
      ft.setStoreTermVectorOffsets(true);
      ft.setStoreTermVectorPositions(true);

      String pathname = "src/test/resources/enwiki-20180820-pages-articles14.xml-p7697599p7744799";
      String languageCode = "en";
      File dump = new File(pathname);
      WikipediaImport wikipediaImport = new WikipediaImport(dump, languageCode, true);
      wikipediaImport.importWikipedia(writer, ft);

      Document doc1 = new Document();
      doc1.add(new Field("title", "riemann bernhard - life and works of bernhard riemann", ft));

      Document doc2 = new Document();
      doc2.add(new Field("title", "thomas bernhard biography - bio and influence in literature", ft));

      Document doc3 = new Document();
      doc3.add(new Field("title", "riemann hypothesis - a deep dive into a mathematical mystery", ft));

      Document doc4 = new Document();
      doc4.add(new Field("title", "bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard", ft));

      writer.addDocument(doc1);
      writer.addDocument(doc2);
      writer.addDocument(doc3);
      writer.addDocument(doc4);

      writer.commit();

      IndexReader reader = DirectoryReader.open(writer);
      String fieldName = "title";
      FieldValuesSentenceIterator fieldValuesSentenceIterator = new FieldValuesSentenceIterator(reader, fieldName);

      Word2Vec vec = new Word2Vec.Builder()
          .layerSize(50)
          .windowSize(6)
          .tokenizerFactory(new DefaultTokenizerFactory())
          .iterate(fieldValuesSentenceIterator)
          .seed(12345)
          .build();

      vec.fit();

      try {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new WordEmbeddingsSimilarity(vec, fieldName, WordEmbeddingsSimilarity.Smoothing.MEAN));
        String queryString = "bernhard riemann influence";

        Terms fieldTerms = MultiFields.getTerms(reader, fieldName);

        INDArray denseAverageTFIDFQueryVector = Nd4j.zeros(vec.getLayerSize());
        Map<String, Double> tfIdfs = new HashMap<>();
        String[] split = queryString.split(" ");
        for (String queryTerm : split) {
          TermsEnum iterator = fieldTerms.iterator();
          BytesRef term;
          while ((term = iterator.next()) != null) {
            TermsEnum.SeekStatus seekStatus = iterator.seekCeil(term);
            if (seekStatus.equals(TermsEnum.SeekStatus.END)) {
              iterator = fieldTerms.iterator();
            }
            if (seekStatus.equals(TermsEnum.SeekStatus.FOUND)) {
              String string = term.utf8ToString();
              if (string.equals(queryTerm)) {
                double tf = iterator.totalTermFreq();
                double docFreq = iterator.docFreq();
                double tfIdf = VectorizeUtils.tfIdf(reader.numDocs(), tf, docFreq);
                tfIdfs.put(string, tfIdf);
              }
            }
          }
          Double n = tfIdfs.get(queryTerm);
          INDArray vector = vec.getLookupTable().vector(queryTerm);
          denseAverageTFIDFQueryVector.addi(vector.div(n));
        }

        INDArray denseAverageQueryVector = vec.getWordVectorsMean(Arrays.asList(split));

        QueryParser parser = new QueryParser(fieldName, new WhitespaceAnalyzer());
        Query query = parser.parse(queryString);

        TopDocs hits = searcher.search(query, 10);
        for (int i = 0; i < hits.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = hits.scoreDocs[i];
          Document doc = searcher.doc(scoreDoc.doc);

          String title = doc.get(fieldName);
          System.out.println(title + " : " + scoreDoc.score);

          Terms docTerms = reader.getTermVector(scoreDoc.doc, fieldName);

          INDArray denseAverageDocumentVector = VectorizeUtils.toDenseAverageVector(docTerms, vec);
          INDArray denseAverageTFIDFDocumentVector = VectorizeUtils.toDenseAverageTFIDFVector(docTerms, reader.numDocs(), vec);

          System.out.println("cosineSimilarityDenseAvg=" + Transforms.cosineSim(denseAverageQueryVector, denseAverageDocumentVector));
          System.out.println("cosineSimilarityDenseAvgTFIDF=" + Transforms.cosineSim(denseAverageTFIDFQueryVector, denseAverageTFIDFDocumentVector));
        }

      } finally {
        WordVectorSerializer.writeWord2VecModel(vec, "target/ch5w2v.zip");
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