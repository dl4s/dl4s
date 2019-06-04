package com.manning.dl4s.ch8;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.BytesRef;

import static org.apache.lucene.search.BooleanClause.Occur.SHOULD;

/**
 * Utility methods for indexing and searching for similar images by feature vectors
 */
public class ImageQueryUtils {

  private static Collection<String> getTokens(Analyzer analyzer, String field, String sampleTextString) throws IOException {
    Collection<String> tokens = new LinkedList<>();
    TokenStream ts = analyzer.tokenStream(field, sampleTextString);
    ts.reset();
    ts.addAttribute(CharTermAttribute.class);
    while (ts.incrementToken()) {
      CharTermAttribute charTermAttribute = ts.getAttribute(CharTermAttribute.class);
      String token = new String(charTermAttribute.buffer(), 0, charTermAttribute.length());
      tokens.add(token);
    }
    ts.end();
    ts.close();
    return tokens;
  }

  private static Query getSimQuery(Analyzer analyzer, String fieldName, String text, int msm) throws IOException {
    return createFingerPrintQuery(fieldName, getTokens(analyzer, fieldName, text), 1, 1.5f);
  }

  public static Query getSimilarityQuery(IndexReader reader, int docId, String similarityField, int msm) {
    try {
      BooleanQuery.Builder similarityQuery = new BooleanQuery.Builder();
      LSHAnalyzer analyzer = new LSHAnalyzer();
      Document doc = reader.document(docId);
      String fvString = doc.get(similarityField);
      if (fvString != null && fvString.trim().length() > 0) {
        Query simQuery = ImageQueryUtils.getSimQuery(analyzer, similarityField, fvString, msm);
        similarityQuery.add(new BooleanClause(simQuery, SHOULD));
      }
      return similarityQuery.build();
    } catch (Exception e) {
      throw new RuntimeException("could not handle similarity query for doc " + docId);
    }
  }

  private static double[] toDoubleArray(byte[] array) {
    int blockSize = Double.SIZE / Byte.SIZE;
    ByteBuffer wrap = ByteBuffer.wrap(array);
    int capacity = array.length / blockSize;
    double[] doubles = new double[capacity];
    for (int i = 0; i < capacity; i++) {
      double e = wrap.getDouble(i * blockSize);
      doubles[i] = e;
    }
    return doubles;
  }

  public static void kNNRerank(int k, double farthestDistance, List<Field> fields, TopDocs docs, IndexSearcher indexSearcher) throws IOException {
    ScoreDoc inputDoc = docs.scoreDocs[0]; // we assume the input doc is the first one returned
    List<Integer> toDiscard = new LinkedList<>();
    for (Field f : fields) {
      String fieldName = f.name();
      BytesRef binaryValue = indexSearcher.doc(inputDoc.doc).getBinaryValue(fieldName);
      double[] inputVector = toDoubleArray(binaryValue.bytes);
      for (int j = 0; j < docs.scoreDocs.length; j++) {
        double[] currentVector = toDoubleArray(indexSearcher.doc(docs.scoreDocs[j].doc)
                .getBinaryValue(fieldName).bytes);
        double distance = dist(inputVector, currentVector) + 1e-10; // constant term to avoid division by zero
        if (distance > farthestDistance || Double.isNaN(distance) || Double.isInfinite(distance)) { // a threshold distance above which current vector is discarded
          toDiscard.add(docs.scoreDocs[j].doc);
        }
        docs.scoreDocs[j].score += 1d / distance; // additive similarity boosting
      }
    }
    if (!toDiscard.isEmpty()) {
      docs.scoreDocs = Arrays.stream(docs.scoreDocs).filter(e -> !toDiscard.contains(e.doc)).toArray(ScoreDoc[]::new); // remove docs that are not close enough
    }
    Arrays.parallelSort(docs.scoreDocs, 0, docs.scoreDocs.length, (o1, o2) -> { // rerank scoreDocs
      return -1 * Double.compare(o1.score, o2.score);
    });
    if (docs.scoreDocs.length > k) {
      docs.scoreDocs = Arrays.copyOfRange(docs.scoreDocs, 0, k); // retain only the top k nearest neighbours
    }
    docs.setMaxScore(docs.scoreDocs[0].score);
    docs.totalHits = k;
  }

  private static double dist(double[] x, double[] y) { // euclidean distance
    double d = 0;
    for (int i = 0; i < x.length; i++) {
      d += Math.pow(y[i] - x[i], 2);
    }
    return Math.sqrt(d);
  }

    private static Query createFingerPrintQuery(String field, Collection<String> minhashes, float similarity, float expectedTruePositive) {
        int bandSize = 1;
        if (expectedTruePositive < 1) {
            bandSize = computeBandSize(minhashes.size(), similarity, expectedTruePositive);
        }

        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        BooleanQuery.Builder childBuilder = new BooleanQuery.Builder();
        int rowInBand = 0;
        for (String minHash : minhashes) {
            TermQuery tq = new TermQuery(new Term(field, minHash));
            if (bandSize == 1) {
                builder.add(new ConstantScoreQuery(tq), BooleanClause.Occur.SHOULD);
            } else {
                childBuilder.add(new ConstantScoreQuery(tq), BooleanClause.Occur.MUST);
                rowInBand++;
                if (rowInBand == bandSize) {
                    builder.add(new ConstantScoreQuery(childBuilder.build()),
                            BooleanClause.Occur.SHOULD);
                    childBuilder = new BooleanQuery.Builder();
                    rowInBand = 0;
                }
            }
        }
        if (childBuilder.build().clauses().size() > 0) {
            for (String token : minhashes) {
                TermQuery tq = new TermQuery(new Term(field, token));
                childBuilder.add(new ConstantScoreQuery(tq), BooleanClause.Occur.MUST);
                rowInBand++;
                if (rowInBand == bandSize) {
                    builder.add(new ConstantScoreQuery(childBuilder.build()),
                            BooleanClause.Occur.SHOULD);
                    break;
                }
            }
        }

        if (expectedTruePositive >= 1.0 && similarity < 1) {
            builder.setMinimumNumberShouldMatch((int) (Math.ceil(minhashes.size() * similarity)));
        }
        return builder.build();

    }

    static int computeBandSize(int numHash, double similarity, double expectedTruePositive) {
        for (int bands = 1; bands <= numHash; bands++) {
            int rowsInBand = numHash / bands;
            double truePositive = 1 - Math.pow(1 - Math.pow(similarity, rowsInBand), bands);
            if (truePositive > expectedTruePositive) {
                return rowsInBand;
            }
        }
        return 1;
    }

}
