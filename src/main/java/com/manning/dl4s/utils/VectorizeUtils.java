/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.manning.dl4s.utils;

import java.io.IOException;
import java.util.Collection;
import java.util.LinkedList;

import com.manning.dl4s.ch5.WordEmbeddingsSimilarity;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * utility class for converting Lucene {@link org.apache.lucene.document.Document}s to <code>Double</code> vectors.
 */
public class VectorizeUtils {

  private VectorizeUtils() {
    // no public constructors
  }

  /**
   * create a sparse <code>Double</code> vector given doc and field term vectors using TF-IDF
   *
   * @param docTerms   term vectors for a given document
   * @param fieldTerms field term vectors
   * @return a TF-IDF sparse vector
   * @throws IOException in case accessing the underlying index fails
   */
  public static double[] toSparseTFIDFDoubleArray(Terms docTerms, Terms fieldTerms, double n) throws IOException {
    TermsEnum fieldTermsEnum = fieldTerms.iterator();
    double[] tfIdfVector = null;
    if (docTerms != null && fieldTerms.size() > -1) {
      tfIdfVector = new double[(int) fieldTerms.size()];
      int i = 0;
      TermsEnum docTermsEnum = docTerms.iterator();
      BytesRef term;
      while ((term = fieldTermsEnum.next()) != null) {
        TermsEnum.SeekStatus seekStatus = docTermsEnum.seekCeil(term);
        if (seekStatus.equals(TermsEnum.SeekStatus.END)) {
          docTermsEnum = docTerms.iterator();
        }
        if (seekStatus.equals(TermsEnum.SeekStatus.FOUND)) {
          long termFreq = docTermsEnum.totalTermFreq();
          int docFreq = fieldTermsEnum.docFreq();
          double tfIdf = tfIdf(n, termFreq, docFreq);
          tfIdfVector[i] = tfIdf;
        } else {
          tfIdfVector[i] = 0d;
        }
        i++;
      }
    }
    return tfIdfVector;
  }

  public static INDArray toDenseAverageTFIDFVector(Terms docTerms, double n, Word2Vec word2Vec) throws IOException {
    INDArray vector = Nd4j.zeros(word2Vec.getLayerSize());
    if (docTerms != null) {
      TermsEnum docTermsEnum = docTerms.iterator();
      BytesRef term;
      while ((term = docTermsEnum.next()) != null) {
        long termFreq = docTermsEnum.totalTermFreq();
        int docFreq = docTermsEnum.docFreq();
        double tfIdf = tfIdf(n, termFreq, docFreq);
        INDArray wordVector = word2Vec.getLookupTable().vector(term.utf8ToString()).div(tfIdf);
        vector.addi(wordVector);
      }
    }
    return vector;
  }

  public static INDArray toDenseAverageVector(Terms docTerms, double n, Word2Vec word2Vec, WordEmbeddingsSimilarity.Smoothing smoothing) throws IOException {
    INDArray documentVector = Nd4j.zeros(word2Vec.getLayerSize());
    if (docTerms != null) {
      TermsEnum docTermsEnum = docTerms.iterator();
      BytesRef term;
      while ((term = docTermsEnum.next()) != null) {
        INDArray wordVector = word2Vec.getLookupTable().vector(term.utf8ToString());
        if (wordVector != null) {
          smoothDocVector(docTerms, n, smoothing, documentVector, docTermsEnum, wordVector);
        }
      }
    }
    return documentVector;
  }

  private static void smoothDocVector(Terms docTerms, double n, WordEmbeddingsSimilarity.Smoothing smoothing,
                                      INDArray documentVector, TermsEnum docTermsEnum, INDArray wordVector) throws IOException {
    double smooth;
    switch (smoothing) {
      case MEAN:
        smooth = docTerms.size();
        break;
      case TF:
        smooth = docTermsEnum.totalTermFreq();
        break;
      case IDF:
        smooth = docTermsEnum.docFreq();
        break;
      case TF_IDF:
        smooth = VectorizeUtils.tfIdf(n, docTermsEnum.totalTermFreq(), docTermsEnum.docFreq());
        break;
      default:
        smooth = VectorizeUtils.tfIdf(n, docTermsEnum.totalTermFreq(), docTermsEnum.docFreq());
    }
    documentVector.addi(wordVector.div(smooth));
  }

  public static INDArray toDenseAverageVector(Terms terms, Word2Vec word2Vec) throws IOException {
    Collection<String> indArrayCollection = new LinkedList<>();
    if (terms != null && terms.size() > -1) {
      TermsEnum docTermsEnum = terms.iterator();
      BytesRef term;
      while ((term = docTermsEnum.next()) != null) {
        TermsEnum.SeekStatus seekStatus = docTermsEnum.seekCeil(term);
        if (seekStatus.equals(TermsEnum.SeekStatus.END)) {
          docTermsEnum = terms.iterator();
        }
        if (seekStatus.equals(TermsEnum.SeekStatus.FOUND)) {
          indArrayCollection.add(term.utf8ToString());
        }

      }
    }
    return word2Vec.getWordVectorsMean(indArrayCollection);
  }

  public static double tfIdf(double n, double termFreq, double docFreq) {
    //weight(term) = (1+log(tf(term)))*log(N/df(term))
    return 1 + Math.log(termFreq) * Math.log(n / docFreq);
  }

  public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (int i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
      normA += Math.pow(vectorA[i], 2);
      normB += Math.pow(vectorB[i], 2);
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  public static INDArray averageWordVectors(Collection<String> words, Word2Vec word2Vec) {
    INDArray denseDocumentVector;
    try {
      denseDocumentVector = word2Vec.getWordVectorsMean(words);
    } catch (Exception e) {
      denseDocumentVector = Nd4j.zeros(word2Vec.getLayerSize());
      int i = 0;
      for (String token : words) {
        INDArray wordVector = word2Vec.getLookupTable().vector(token);
        if (wordVector != null) {
          denseDocumentVector.addi(wordVector);
          i++;
        }
        INDArray unkVector = word2Vec.getLookupTable().vector(word2Vec.getUNK());
        if (unkVector != null) {
          denseDocumentVector.addi(unkVector);
          i++;
        }
      }
      denseDocumentVector.divi(i);
    }
    return denseDocumentVector;
  }
}
