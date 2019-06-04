package com.manning.dl4s.ch7.embeddings;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.Charset;
import java.util.*;

import com.manning.dl4s.ch7.ParallelSentence;
import com.manning.dl4s.ch7.TMXParser;
import com.manning.dl4s.ch7.TranslatorTool;
import com.manning.dl4s.utils.LuceneTokenizerFactory;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.LowerCaseFilterFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizerFactory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Machine translation based on learned source / target word2vec models and a translation matrix.
 * @see <a href="https://arxiv.org/abs/1309.4168">Paper from Mikolov et al.</a>
 */
public class LinearProjectionMTEmbeddings implements TranslatorTool {

    private final Logger log = LoggerFactory.getLogger(getClass());

    private final Word2Vec sourceWord2Vec;
    private final Word2Vec targetWord2Vec;
    private final INDArray translationMatrix;
    private final TokenizerFactory tokenizerFactory;

    public LinearProjectionMTEmbeddings(String sourceW2V, String targetW2V, String tm) throws Exception {
        Analyzer simpleAnalyzer = CustomAnalyzer.builder()
                .withTokenizer(StandardTokenizerFactory.class)
                .addTokenFilter(LowerCaseFilterFactory.class)
                .build();
        tokenizerFactory = new LuceneTokenizerFactory(simpleAnalyzer);
        translationMatrix = Nd4j.read(new FileInputStream(tm));
        sourceWord2Vec = WordVectorSerializer.readWord2VecModel(sourceW2V);
        targetWord2Vec = WordVectorSerializer.readWord2VecModel(targetW2V);
    }

    public LinearProjectionMTEmbeddings(File tmxFile, File dictionaryFile, String source, String target, int layerSize, int epochs) {

        try {

            log.info("parsing tmx file {}", tmxFile);

            Collection<ParallelSentence> parallelSentences = new TMXParser(tmxFile, source, target).parse();
            List<String> strings = FileUtils.readLines(dictionaryFile, Charset.forName("utf-8"));
            String dicseparator = " ";
            int dictionaryLength = strings.size() - 1;

            Analyzer simpleAnalyzer = CustomAnalyzer.builder()
                    .withTokenizer(StandardTokenizerFactory.class)
                    .addTokenFilter(LowerCaseFilterFactory.class)
                    .build();
            tokenizerFactory = new LuceneTokenizerFactory(simpleAnalyzer);

            Collection<String> sources = new LinkedList<>();
            Collection<String> targets = new LinkedList<>();
            for (ParallelSentence sentence : parallelSentences) {
                sources.add(sentence.getSource());
                targets.add(sentence.getTarget());
            }

            log.info("fitting monolingual word embeddings");

            sourceWord2Vec = new Word2Vec.Builder()
                    .iterate(new CollectionSentenceIterator(sources))
                    .tokenizerFactory(tokenizerFactory)
                    .limitVocabularySize(dictionaryLength)
                    .layerSize(layerSize)
                    .epochs(epochs)
                    .build();
            sourceWord2Vec.fit();

            targetWord2Vec = new Word2Vec.Builder()
                    .iterate(new CollectionSentenceIterator(targets))
                    .tokenizerFactory(tokenizerFactory)
                    .limitVocabularySize(dictionaryLength)
                    .layerSize(layerSize)
                    .epochs(epochs)
                    .build();
            targetWord2Vec.fit();

            log.info("filtering embeddings by dictionary");

            int count = 0;
            INDArray sourceVectors = Nd4j.zeros(dictionaryLength, layerSize);
            INDArray targetVectors = Nd4j.zeros(dictionaryLength, layerSize);

            for (String line : strings) {
                String[] pair = line.split(dicseparator);
                String sourceWord = pair[0];
                String targetWord = pair[1];
                if (sourceWord2Vec.hasWord(sourceWord) && targetWord2Vec.hasWord(targetWord)) {
                    sourceVectors.putRow(count, sourceWord2Vec.getWordVectorMatrix(sourceWord));
                    targetVectors.putRow(count, targetWord2Vec.getWordVectorMatrix(targetWord));
                    count++;
                }
            }

            log.info("building translation projection matrix");

            INDArray pseudoInverseSourceMatrix = InvertMatrix.pinvert(sourceVectors, false);

            log.debug("inverse of source matrix is {}", Arrays.toString(pseudoInverseSourceMatrix.shape()));

            translationMatrix = pseudoInverseSourceMatrix.mmul(targetVectors).transpose();

            log.info("translation matrix and word embeddings build finished");

            WordVectorSerializer.writeWord2VecModel(sourceWord2Vec, "sw2v");
            WordVectorSerializer.writeWord2VecModel(targetWord2Vec, "tw2v");
            Nd4j.write(new FileOutputStream("tm"), translationMatrix);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public List<Translation> decodeWord(int n, String sourceWord) {
        if (sourceWord2Vec.hasWord(sourceWord)) {
            INDArray sourceWordVector = sourceWord2Vec.getWordVectorMatrix(sourceWord);
            INDArray targetVector = sourceWordVector.mmul(translationMatrix.transpose());
            Collection<String> strings = targetWord2Vec.wordsNearest(targetVector, n);
            List<Translation> translations = new ArrayList<>(strings.size());
            for (String s : strings) {
                Translation t = new Translation(s, targetWord2Vec.similarity(s, sourceWord));
                translations.add(t);
                log.info("added translation {} for {}", t, sourceWord);
            }
            return translations;
        } else {
            return Collections.emptyList();
        }
    }

    @Override
    public Collection<Translation> translate(String text) {
        StringBuilder stringBuilder = new StringBuilder();
        double score = 0;
        List<String> tokens = tokenizerFactory.create(text).getTokens();
        for (String t : tokens) {
            if (stringBuilder.length() > 0) {
                stringBuilder.append(' ');
            }
            List<Translation> translations = decodeWord(1, t);
            Translation translation = translations.get(0);
            score += translation.getScore();
            stringBuilder.append(translation);
        }
        String string = stringBuilder.toString();
        Translation translation = new Translation(string, score / (double) tokens.size());
        log.info("{} translated into {}", text, translation);
        return Collections.singletonList(translation);
    }
}
