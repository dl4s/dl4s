package com.manning.dl4s.ch9.functions;

import java.util.Collection;
import java.util.LinkedList;
import java.util.Map;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;

public class ResultTransformer implements MapFunction<Map<String, String[]>, Collection<Tuple2<String,String>>> {
  @Override
  public Collection<Tuple2<String,String>> map(Map<String, String[]> value) throws Exception {
    Collection<Tuple2<String,String>> res = new LinkedList<>();

    for (Map.Entry<String, String[]> entry : value.entrySet()) {
      StringBuilder resBuffer = new StringBuilder();
      for (String r : entry.getValue()) {
        if (resBuffer.length() > 0) {
          resBuffer.append('\n');
        }
        resBuffer.append(r);
      }
      res.add(new Tuple2<>(entry.getKey(), resBuffer.toString()));
    }
    return res;
  }
}
