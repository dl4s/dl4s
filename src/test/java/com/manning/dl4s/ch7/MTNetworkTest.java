package com.manning.dl4s.ch7;

import com.manning.dl4s.ch7.encdec.MTNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

import static org.junit.Assert.assertNotNull;

/**
 * Tests for {@link MTNetwork}
 */
public class MTNetworkTest {

    @BeforeClass
    public static void setupBefore() throws Exception {
        String filePath = "src/test/resources/en-it.tmx.gz";
        if (!new File(filePath).exists()) {
            URI uri = URI.create("https://object.pouta.csc.fi/OPUS-EMEA/v3/tmx/en-it.tmx.gz");
            InputStream in = uri.toURL().openStream();
            Files.copy(in, Paths.get(filePath));
            in.close();
        }
        GZIPInputStream gzis =
                new GZIPInputStream(new FileInputStream(filePath));
        FileOutputStream out =
                new FileOutputStream("src/test/resources/en-it_emea.tmx");
        byte[] buffer = new byte[1024];
        int len;
        while ((len = gzis.read(buffer)) > 0) {
            out.write(buffer, 0, len);
        }

        gzis.close();
        out.close();
    }

    @Test
    public void testTraining() throws Exception {

        String[] testQueries = new String[] {"it has been proposed", "identity card validity in the UK", "other ingredients",
        "absorption profile is due to the product", "container must not be refilled"};

        String tmxFile = "src/test/resources/en-it_emea.tmx";
        String sourceCode = "en";
        String targetCode = "it";

        MTNetwork mtNetwork = new MTNetwork(tmxFile, sourceCode, targetCode, new ScoreIterationListener(1000));
        for (String testString : testQueries) {
            assertNotNull(mtNetwork.output(testString, 0));
        }

    }

}