package com.manning.dl4s.ch7;

import com.manning.dl4s.ch7.embeddings.LinearProjectionMTEmbeddings;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.zip.GZIPInputStream;

import static org.junit.Assert.assertNotNull;

/**
 * Tests for {@link LinearProjectionMTEmbeddings}
 */
public class LinearProjectionMTEmbeddingsTest {

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
  public void testEmbeddingsTrainAndInference() throws Exception {
    String[] ts = new String[]{"disease", "cure", "current", "latest", "day", "delivery", "destroy",
        "design", "enoxacine", "other", "validity", "other ingredients", "absorption profile",
            "container must not be refilled"};
    File tmxFile = new File("src/test/resources/en-it_emea.tmx");
    File dictionaryFile = new File("src/test/resources/en-it_emea.dic");
    LinearProjectionMTEmbeddings linearProjectionMTEmbeddings = new LinearProjectionMTEmbeddings(tmxFile, dictionaryFile, "en", "it", 60, 1);

    for (String t : ts) {
      Collection<TranslatorTool.Translation> translations = linearProjectionMTEmbeddings.translate(t);
      assertNotNull(translations);
    }
  }
}