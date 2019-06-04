package com.manning.dl4s.ch3;

import java.util.Collection;

import com.manning.dl4s.utils.encdec.EncoderDecoderLSTM;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lucene {@link QueryParser} generating alternative underlying queries by sampling from a seq2seq {@link EncoderDecoderLSTM}.
 */
public class Seq2SeqAltQueriesQueryParser extends QueryParser {

  private final Logger log = LoggerFactory.getLogger(getClass());

  private final EncoderDecoderLSTM net;

  public Seq2SeqAltQueriesQueryParser(String field, Analyzer a, EncoderDecoderLSTM net) {
    super(field, a);
    this.net = net;
  }

  @Override
  public Query parse(String query) throws ParseException {
    BooleanQuery.Builder builder = new BooleanQuery.Builder();
    builder.add(new BooleanClause(super.parse(query), BooleanClause.Occur.MUST));

    Collection<String> samples = net.output(query);

    StringBuilder alternativeQuery = new StringBuilder();
    for (String sample : samples) {
      if (alternativeQuery.length() > 0) {
        alternativeQuery.append(' ');
      }
      alternativeQuery.append(sample);
      log.debug("{} -> {}", query,  sample);
    }
    builder.add(new BooleanClause(super.parse(alternativeQuery.toString()), BooleanClause.Occur.SHOULD));

    return builder.build();
  }

}
