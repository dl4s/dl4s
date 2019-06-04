package com.manning.dl4s.ch4;

import com.google.common.collect.Sets;
import com.manning.dl4s.utils.CharacterIterator;
import org.apache.lucene.search.spell.Dictionary;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

/**
 * An extension of {@link CharLSTMNeuralLookup} which uses {@link Word2Vec} to provide more diverse suggestions by
 * substituting word2vec nearest words having a similarity greater than 0.7.
 */
public class CharLSTMWord2VecLookup extends CharLSTMNeuralLookup {

    private final Logger log = LoggerFactory.getLogger(getClass());

    private Word2Vec word2Vec;

    public CharLSTMWord2VecLookup(MultiLayerNetwork net, CharacterIterator iter, Word2Vec word2Vec) {
        super(net, iter);
        this.word2Vec = word2Vec;
    }

    public CharLSTMWord2VecLookup(int lstmLayerSize, int miniBatchSize, int exampleLength, int tbpttLength, int numEpochs,
                                  int noOfHiddenLayers, WeightInit weightInit, IUpdater updater, Activation activation,
                                  DocumentIterator iterator) {
        super(lstmLayerSize, miniBatchSize, exampleLength, tbpttLength, numEpochs, noOfHiddenLayers, weightInit, updater,
                activation);
        this.word2Vec = new Word2Vec.Builder()
                .iterate(iterator)
                .layerSize(lstmLayerSize)
                .build();
        word2Vec.fit();
    }

    @Override
    public void build(InputIterator inputIterator) throws IOException {
        super.build(inputIterator);
        initW2V();
    }

    private void initW2V() {
        if (word2Vec == null) {
            String textFilePath = characterIterator.getTextFilePath();
            log.info("reading sequences from {}", textFilePath);
            this.word2Vec = new Word2Vec.Builder()
                    .epochs(5)
                    .layerSize(200)
                    .tokenizerFactory(new DefaultTokenizerFactory())
                    .iterate(new FileDocumentIterator(textFilePath))
                    .build();
            this.word2Vec.fit();
        }
    }

    @Override
    public List<LookupResult> lookup(CharSequence key, Set<BytesRef> contexts, boolean onlyMorePopular, int num) throws IOException {
        Set<LookupResult> results = Sets.newCopyOnWriteArraySet(super.lookup(key, contexts, onlyMorePopular, num));
        for (LookupResult lr : results) {
            String suggestionString = lr.key.toString();
            for (String word : word2Vec.getTokenizerFactory().create(suggestionString).getTokens()) {
                Collection<String> nearestWords = word2Vec.wordsNearest(word, 1);
                for (String nearestWord : nearestWords) {
                    if (word2Vec.similarity(word, nearestWord) > 0.7) {
                        results.addAll(enhanceSuggestion(lr, word, nearestWord));
                    }
                }
            }
        }
        return new ArrayList<>(results);
    }

    private Collection<LookupResult> enhanceSuggestion(LookupResult lr, String word, String nearestWord) {
        return Collections.singletonList(new LookupResult(lr.key.toString().replace(word, nearestWord), (long) (lr.value * 0.7)));
    }
}
