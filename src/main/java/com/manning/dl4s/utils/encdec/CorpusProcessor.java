package com.manning.dl4s.utils.encdec;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Slightly adapted version of DL4J's {@link CorpusProcessor}
 *
 * @see <a href="https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/encdec/CorpusProcessor"></a>
 */
public class CorpusProcessor {
    public static final String SPECIALS = "!\"#$;%^:?*()[]{}<>«»,.–—=+…";
    private Set<String> dictSet = new HashSet<>();
    private Map<String, Double> freq = new HashMap<>();
    private Map<String, Double> dict = new HashMap<>();
    private boolean countFreq;
    private InputStream is;
    private int rowSize;

    public CorpusProcessor(String filename, int rowSize, boolean countFreq) throws FileNotFoundException {
        this(new FileInputStream(filename), rowSize, countFreq);
    }

    public CorpusProcessor(InputStream is, int rowSize, boolean countFreq) {
        this.is = is;
        this.rowSize = rowSize;
        this.countFreq = countFreq;
    }

    public void start() throws IOException {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                processLine(line);
            }
        }
    }

    public void processLine(String lastLine) {
        tokenizeLine(lastLine, dictSet, false);
    }

    // here we not only split the words but also store punctuation marks
    public void tokenizeLine(String lastLine, Collection<String> resultCollection, boolean addSpecials) {
        String[] words = lastLine.split("[ \t]");
        for (String word : words) {
            if (!word.isEmpty()) {
                boolean specialFound = true;
                while (specialFound && !word.isEmpty()) {
                    for (int i = 0; i < word.length(); ++i) {
                        int idx = SPECIALS.indexOf(word.charAt(i));
                        specialFound = false;
                        if (idx >= 0) {
                            String word1 = word.substring(0, i);
                            if (!word1.isEmpty()) {
                                addWord(resultCollection, word1);
                            }
                            if (addSpecials) {
                                addWord(resultCollection, String.valueOf(word.charAt(i)));
                            }
                            word = word.substring(i + 1);
                            specialFound = true;
                            break;
                        }
                    }
                }
                if (!word.isEmpty()) {
                    addWord(resultCollection, word);
                }
            }
        }
    }

    private void addWord(Collection<String> coll, String word) {
        if (coll != null) {
            coll.add(word);
        }
        if (countFreq) {
            Double count = freq.get(word);
            if (count == null) {
                freq.put(word, 1.0);
            } else {
                freq.put(word, count + 1);
            }
        }
    }

    public Map<String, Double> getFreq() {
        return freq;
    }

    public void setDict(Map<String, Double> dict) {
        this.dict = dict;
    }

    /**
     * Converts an iterable sequence of words to a list of indices. This will
     * never return {@code null} but may return an empty {@link List}.
     *
     * @param words
     *            sequence of words
     * @return list of indices.
     */
    public final List<Double> wordsToIndexes(final Iterable<String> words) {
        int i = rowSize;
        final List<Double> wordIdxs = new LinkedList<>();
        for (final String word : words) {
            if (--i == 0) {
                break;
            }
            final Double wordIdx = dict.get(word);
            if (wordIdx != null) {
                wordIdxs.add(wordIdx);
            } else {
                wordIdxs.add(0.0);
            }
        }
        return wordIdxs;
    }

}
