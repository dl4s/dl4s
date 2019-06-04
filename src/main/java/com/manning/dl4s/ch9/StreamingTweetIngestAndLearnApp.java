package com.manning.dl4s.ch9;

import com.manning.dl4s.ch9.functions.*;
import com.twitter.hbc.core.endpoint.StatusesFilterEndpoint;
import com.twitter.hbc.core.endpoint.StreamingEndpoint;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.shaded.guava18.com.google.common.collect.Lists;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Serializable;

import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Streaming app for ingestion of tweets and related learning / update functions
 */
public class StreamingTweetIngestAndLearnApp {

  private static final Logger log = LoggerFactory.getLogger(StreamingTweetIngestAndLearnApp.class);

  public static void main(String[] args) throws Exception {

    final StreamExecutionEnvironment env =
        StreamExecutionEnvironment.getExecutionEnvironment().setParallelism(1);

    env.getConfig().enableObjectReuse();
    env.setStreamTimeCharacteristic(TimeCharacteristic.IngestionTime);

    Properties props = new Properties();
    props.load(StreamingTweetIngestAndLearnApp.class.getResourceAsStream("/twitter.properties"));
    TwitterSource twitterSource = new TwitterSource(props);
    String[] tags = {"neural search", "natural language processing", "lucene", "deep learning", "word embeddings", "manning"};
    twitterSource.setCustomEndpointInitializer(new FilterEndpoint(tags));
    log.info("ingesting tweets for tags {}", Arrays.toString(tags));

    DataStream<Tweet> twitterStream = env.addSource(twitterSource)
        .filter((FilterFunction<String>) value -> value.contains("created_at"))
        .flatMap(new TweetJsonConverter());

    int batchSize = 5;

    twitterStream
          .countWindowAll(batchSize)
          .apply(new ModelAndIndexUpdateFunction())
          .map(new MultiRetrieverFunction())
          .map(new ResultTransformer()).countWindowAll(1)
          .apply(new TupleEvictorFunction())
          .writeAsCsv("src/main/html/data.csv", FileSystem.WriteMode.OVERWRITE);
    env.execute();
  }

  static class FilterEndpoint implements TwitterSource.EndpointInitializer, Serializable {

    private final List<String> tags;

    FilterEndpoint(final String... tags) {
      this.tags = Lists.newArrayList(tags);
    }

    @Override
    public StreamingEndpoint createEndpoint() {
      StatusesFilterEndpoint ep = new StatusesFilterEndpoint();
      ep.trackTerms(tags);
      return ep;
    }
  }


}
