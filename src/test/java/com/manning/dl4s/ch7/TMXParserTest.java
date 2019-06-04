package com.manning.dl4s.ch7;

import org.apache.commons.io.IOUtils;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.LinkedList;
import java.util.zip.GZIPInputStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Tests for {@link TMXParser}
 */
public class TMXParserTest {

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
  public void testParsing() throws Exception {
    TMXParser tmxParser = new TMXParser(Paths.get("src/test/resources/en-it_emea.tmx").toFile(), "it", "en");
    Collection<ParallelSentence> parse = tmxParser.parse();
    assertNotNull(parse);
    assertEquals(361089, parse.size());

    FileOutputStream fos = new FileOutputStream("target/pc.txt");
    Collection<String> lines = new LinkedList<>();
    for (ParallelSentence ps : parse) {
      lines.add(ps.getSource());
      lines.add(ps.getTarget());
    }
    IOUtils.writeLines(lines,"\n",fos, Charset.forName("utf-8"));
    fos.flush();
    fos.close();

  }
}