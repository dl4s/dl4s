package com.manning.dl4s.ch3;

import java.util.Collection;

import com.manning.dl4s.utils.CharacterIterator;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.manning.dl4s.utils.NeuralNetworksUtils.sampleFromNetwork;

/**
 * Lucene {@link QueryParser} generating alternative underlying queries by sampling from a {@link MultiLayerNetwork}
 * (e.g. charLSTM).
 */
public class AltQueriesQueryParser extends QueryParser {

  private final Logger log = LoggerFactory.getLogger(getClass());

  private final MultiLayerNetwork rnn;
  private final CharacterIterator characterIterator;

  public AltQueriesQueryParser(String field, Analyzer a, MultiLayerNetwork rnn, CharacterIterator characterIterator) {
    super(field, a);
    this.rnn = rnn;
    this.characterIterator = characterIterator;
  }

  @Override
  public Query parse(String query) throws ParseException {
    BooleanQuery.Builder builder = new BooleanQuery.Builder();
    builder.add(new BooleanClause(super.parse(query), BooleanClause.Occur.MUST));

    Collection<String> samples = sampleFromNetwork(rnn, characterIterator, query + "\n", 3, '\n').keySet();

    for (String sample : samples) {
      if (sample.contains("\n")) {
        sample = sample.substring(sample.indexOf("\n"));
      }

      log.debug("{} -> {}", query,  sample);
      builder.add(new BooleanClause(super.parse(sample), BooleanClause.Occur.SHOULD));
    }

    return builder.build();
  }

}
