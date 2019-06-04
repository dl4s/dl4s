/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.manning.dl4s.utils;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;
import javax.xml.transform.stream.StreamSource;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class to import Wikipedia dumps into a Lucene index.
 */
public class WikipediaImport {

  private final Logger log = LoggerFactory.getLogger(getClass());

  private static final Pattern pattern = Pattern.compile("\\[Category:(\\w+([|\\s']\\w*)*)]");
  public static final String CATEGORY = "category";
  public static final String TEXT = "text";
  public static final String TITLE = "title";
  public static final String LANG = "lang";
  private final File dump;
  private final boolean doReport;
  private final String languageCode;

  public WikipediaImport(File dump, String languageCode, boolean doReport) {
    this.dump = dump;
    this.languageCode = languageCode;
    this.doReport = doReport;
  }

  public void importWikipedia(IndexWriter indexWriter, FieldType ft) throws Exception {
    long start = System.currentTimeMillis();
    int count = 0;
    if (doReport) {
      log.info("Importing {}...", dump);
    }

    String title = null;
    String text = null;
    Set<String> cats = new HashSet<>();
    XMLInputFactory factory = XMLInputFactory.newInstance();
    StreamSource source;
    source = new StreamSource(dump);
    XMLStreamReader reader = factory.createXMLStreamReader(source);
    while (reader.hasNext()) {
      switch (reader.next()) {
        case XMLStreamConstants.START_ELEMENT:
          if ("title".equals(reader.getLocalName())) {
            title = reader.getElementText();
          } else if ("text".equals(reader.getLocalName())) {
            text = reader.getElementText();
            Matcher matcher = pattern.matcher(text);
            int pos = 0;
            while (matcher.find(pos)) {
              String group = matcher.group(1);
              String catName = group.replaceAll("\\|\\s", "").replaceAll("\\|\\*", "");
              Collections.addAll(cats, catName.split("\\|"));
              pos = matcher.end();
            }
          }
          break;
        case XMLStreamConstants.END_ELEMENT:
          if ("page".equals(reader.getLocalName())) {
            Document page = new Document();
            page.add(new Field(TITLE, title, ft));
            page.add(new Field(TEXT, text, ft));
            for (String cat : cats) {
              page.add(new Field(CATEGORY, cat, ft));
            }
            page.add(new StringField(LANG, languageCode, Field.Store.YES));
            indexWriter.addDocument(page);
            count++;
            if (count % 10000 == 0) {
              batchDone(indexWriter, start, count);
            }
            cats.clear();
          }
          break;
      }
    }

    indexWriter.commit();

    if (doReport) {
      long millis = System.currentTimeMillis() - start;
      log.info(
          "Imported {} pages in {} seconds ({}ms/page)",
          count, millis / 1000, (double) millis / count);
    }

  }

  protected void batchDone(IndexWriter indexWriter, long start, int count) throws IOException {
    indexWriter.commit();
    if (doReport) {
      long millis = System.currentTimeMillis() - start;
      log.info(
          "Added {} pages in {} seconds ({}ms/page)",
          count, millis / 1000, (double) millis / count);
    }
  }

}