package com.manning.dl4s.ch7;

import java.util.Collection;
import java.util.Map;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.langdetect.Language;
import opennlp.tools.langdetect.LanguageDetector;

/**
 * A {@link QueryParser} that takes the original user entered query,
 * recognizes the language via openNLP's {@link LanguageDetector}, picks up the {@link TranslatorTool}s that
 * can translate from the given language, perform decoding (translation), parses the translated
 * queries and then adds them as optional queries in a {@link BooleanQuery}.
 */
public class MTQueryParser extends QueryParser {

  private final Logger log = LoggerFactory.getLogger(getClass());

  private final LanguageDetector languageDetector;
  private final Map<String, Collection<TranslatorTool>> decoderMappings;

  MTQueryParser(String f, Analyzer a, LanguageDetector languageDetector,
                Map<String, Collection<TranslatorTool>> decoderMappings) {
    super(f, a);
    this.languageDetector = languageDetector;
    this.decoderMappings = decoderMappings;
  }

  @Override
  public Query parse(String query) throws ParseException {

    BooleanQuery.Builder builder = new BooleanQuery.Builder();

    // add user entered query
    builder.add(new BooleanClause(super.parse(query), BooleanClause.Occur.SHOULD));

    // perform language detection
    Language language = languageDetector.predictLanguage(query);
    String languageString = language.getLang();
    if (languageString == null) {
      languageString = "eng";
    }

    log.info("detected language {} for query '{}'", languageString, query);

    // find which mt model supports the extracted language
    Collection<TranslatorTool> decoders = decoderMappings.get(languageString);

    if (decoders == null) {
      decoders = decoderMappings.get("eng"); // use default en->xyz decoders
      if (decoders == null) {
        decoders = decoderMappings.get("ita");
      }
    }

    // perform query translation for each of the mt models
    for (TranslatorTool d : decoders) {
      Collection<TranslatorTool.Translation> translations = d.translate(query);

      log.info("found {} translations", translations.size());

      // create and bind corresponding queries
      for (TranslatorTool.Translation translation : translations) {
        log.info("query {} translated into {}", query, translation);
        String translationString = translation.getTranslationString();
        builder.add(new BooleanClause(super.parse(translationString), BooleanClause.Occur.SHOULD));
      }
    }

    return builder.build();
  }

}
