package com.manning.dl4s.ch9.functions;

import java.util.Collection;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow;
import org.apache.flink.util.Collector;

public class TupleEvictorFunction implements AllWindowFunction<Collection<Tuple2<String,String>>, Tuple2<String,String>, GlobalWindow> {

  @Override
  public void apply(GlobalWindow globalWindow, Iterable<Collection<Tuple2<String,String>>> iterable, Collector<Tuple2<String,String>> collector) throws Exception {
    for (Collection<Tuple2<String,String>> c : iterable) {
      for (Tuple2<String,String> t : c) {
        collector.collect(t);
      }
    }
  }
}
