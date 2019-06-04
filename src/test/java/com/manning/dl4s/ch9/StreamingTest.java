package com.manning.dl4s.ch9;

import org.junit.Ignore;
import org.junit.Test;

import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Tests for streaming indexing and learning
 */
public class StreamingTest {

    @Ignore("add twitter tokens credentials in twitter.properties")
    @Test
    public void testStreamingApps() throws Exception {
        ExecutorService executorService = Executors.newSingleThreadExecutor();
        Collection<Callable<Object>> callables = new LinkedList<>();
        callables.add(() -> {
            StreamingTweetIngestAndLearnApp.main(new String[0]);
            return null;
        });
        callables.add(() -> {
            OutputApp.main(new String[0]);
            return null;
        });
        executorService.invokeAll(callables, 1, TimeUnit.MINUTES);
    }

}