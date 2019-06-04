package com.manning.dl4s.ch7;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;

import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

/**
 * Parser for TMX files
 */
public class TMXParser {

  private static final String ELEMENT_TU = "tu";
  private static final String ELEMENT_TUV = "tuv";
  private static final String ELEMENT_SEG = "seg";

  private final File tmxFile;
  private final String sourceCode;
  private final String targetCode;
  private final XMLInputFactory factory = XMLInputFactory.newInstance();
  private final Collection<ParallelSentence> parallelSentenceCollection;

  public TMXParser(final File tmxFile, String sourceCode, String targetCode) {
    this.tmxFile = tmxFile;
    this.sourceCode = sourceCode;
    this.targetCode = targetCode;
    this.parallelSentenceCollection = new LinkedList<>();
  }

  public Collection<ParallelSentence> parse() throws IOException, XMLStreamException {
    try (final InputStream stream = new FileInputStream(tmxFile)) {
      final XMLEventReader reader = factory.createXMLEventReader(stream);
      while (reader.hasNext()) {
        final XMLEvent event = reader.nextEvent();
        if (event.isStartElement() && event.asStartElement().getName()
            .getLocalPart().equals(ELEMENT_TU)) {
          parseTU(reader);
        }
      }
    }
    return Collections.unmodifiableCollection(parallelSentenceCollection);
  }

  private void parseTU(final XMLEventReader reader) throws XMLStreamException {
    String source = null;
    String target = null;
    String code = null;

    while (reader.hasNext()) {
      final XMLEvent event = reader.nextEvent();
      if (event.isEndElement() && event.asEndElement().getName().getLocalPart().equals(ELEMENT_TU)) {
        if (source != null && target != null) {
          ParallelSentence sentence = new ParallelSentence(source, target);
          parallelSentenceCollection.add(sentence);
        }
        return;
      }
      if (event.isStartElement()) {
        final StartElement element = event.asStartElement();
        final String elementName = element.getName().getLocalPart();
        switch (elementName) {
          case ELEMENT_TUV:
            Iterator attributes = element.getAttributes();
            while(attributes.hasNext()) {
              Attribute next = (Attribute) attributes.next();
              code = next.getValue();
            }
            break;
          case ELEMENT_SEG:
            if (sourceCode.equals(code)) {
              source = reader.getElementText();
            } else if (targetCode.equals(code)) {
              target = reader.getElementText();
            }
            break;
        }
      }
    }

  }


}