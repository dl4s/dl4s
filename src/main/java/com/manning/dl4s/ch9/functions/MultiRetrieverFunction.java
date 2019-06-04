package com.manning.dl4s.ch9.functions;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.manning.dl4s.ch5.WordEmbeddingsSimilarity;
import com.manning.dl4s.ch6.ParagraphVectorsSimilarity;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Retrieve function using different ranking models
 */
public class MultiRetrieverFunction implements MapFunction<Long, Map<String, String[]>> {

  private final Logger log = LoggerFactory.getLogger(getClass());

  @Override
  public Map<String, String[]> map(Long commit) throws Exception {

    Path indexPath = Utils.getIndexPath();
    ParagraphVectors paragraphVectors = Utils.fetchParagraghVectors(indexPath);
    Word2Vec word2Vec = Utils.fetchWordVectors(indexPath);

    String fieldName = "text";

    Map<String, IndexSearcher> searchers = new HashMap<>();
    FSDirectory directory = FSDirectory.open(indexPath);

    IndexReader reader1 = DirectoryReader.open(directory);
    IndexSearcher classic = new IndexSearcher(reader1);
    classic.setSimilarity(new ClassicSimilarity());
    searchers.put("classic", classic);

    IndexReader reader2 = DirectoryReader.open(directory);
    IndexSearcher bm25 = new IndexSearcher(reader2);
    bm25.setSimilarity(new BM25Similarity());
    searchers.put("bm25", bm25);

    IndexReader reader3 = DirectoryReader.open(directory);
    IndexSearcher pv = new IndexSearcher(reader3);
    paragraphVectors.setTokenizerFactory(new DefaultTokenizerFactory());
    pv.setSimilarity(new ParagraphVectorsSimilarity(paragraphVectors, fieldName));
    searchers.put("document embedding ranking", pv);

    IndexReader reader4 = DirectoryReader.open(directory);
    IndexSearcher lmd = new IndexSearcher(reader4);
    lmd.setSimilarity(new LMDirichletSimilarity());
    searchers.put("language model dirichlet", lmd);

    IndexReader reader5 = DirectoryReader.open(directory);
    IndexSearcher wv = new IndexSearcher(reader5);
    pv.setSimilarity(new WordEmbeddingsSimilarity(word2Vec, fieldName, WordEmbeddingsSimilarity.Smoothing.TF_IDF));
    searchers.put("average word embedding ranking", wv);

    Map<String, String[]> results = new HashMap<>();
    int topK = 1;
    QueryParser simpleQueryParser = new QueryParser(fieldName, new StandardAnalyzer());
    String queryText = "deep learning search";
    for (Map.Entry<String, IndexSearcher> entry : searchers.entrySet()) {
      Query query = simpleQueryParser.parse(queryText);
      log.debug("running query '{}' for {}", query.toString(), entry.getKey());
      IndexSearcher searcher = entry.getValue();
      TopDocs topDocs = searcher.search(query, topK);
      String[] stringResults = new String[topK];
      int i = 0;
      for (ScoreDoc sd : topDocs.scoreDocs) {
        Document doc = searcher.doc(sd.doc);
        IndexableField text = doc.getField(fieldName);
        if (text != null) {
          stringResults[i] = text.stringValue().replaceAll(",","");
        }
        i++;
      }
      results.put(entry.getKey(), stringResults);
      log.info("{} - {}", entry.getKey(), Arrays.toString(stringResults));
    }

    reader1.close();
    reader2.close();
    reader3.close();
    reader4.close();
    directory.close();
    return results;
  }
}
