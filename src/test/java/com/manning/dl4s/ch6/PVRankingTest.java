package com.manning.dl4s.ch6;

import com.google.common.io.Files;
import com.manning.dl4s.ch5.FieldValuesLabelAwareIterator;
import com.manning.dl4s.utils.WikipediaImport;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.Collections;

/**
 * Tests for ranking based on {@link ParagraphVectorsSimilarity}
 */
public class PVRankingTest {

    @Test
    public void testRankingWithParagraphVectors() throws Exception {
        Path path = Paths.get(Files.createTempDir().toURI());
        Directory directory = FSDirectory.open(path);

        try {

            IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());

            IndexWriter writer = new IndexWriter(directory, config);
            FieldType ft = new FieldType(TextField.TYPE_STORED);
            ft.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
            ft.setTokenized(true);
            ft.setStored(true);
            String fieldName = "title";

            String pathname = "src/test/resources/enwiki-20180820-pages-articles14.xml-p7697599p7744799";
            String languageCode = "en";

            File dump = new File(pathname);
            WikipediaImport wikipediaImport = new WikipediaImport(dump, languageCode, true);
            wikipediaImport.importWikipedia(writer, ft);

            Document doc1 = new Document();
            doc1.add(new Field(fieldName, "riemann bernhard - life and works of bernhard riemann", ft));

            Document doc2 = new Document();
            doc2.add(new Field(fieldName, "thomas bernhard biography - bio and influence in literature", ft));

            Document doc3 = new Document();
            doc3.add(new Field(fieldName, "riemann hypothesis - a deep dive into a mathematical mystery", ft));

            Document doc4 = new Document();
            doc4.add(new Field(fieldName, "bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard", ft));

            writer.addDocument(doc1);
            writer.addDocument(doc2);
            writer.addDocument(doc3);
            writer.addDocument(doc4);
            writer.commit();

            IndexReader reader = DirectoryReader.open(writer);

            FieldValuesLabelAwareIterator iterator = new FieldValuesLabelAwareIterator(reader, fieldName);

            ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                    .iterate(iterator)
                    .layerSize(50)
                    .seed(12345)
                    .tokenizerFactory(new DefaultTokenizerFactory())
                    .build();

            paragraphVectors.fit();

            String queryString = "bernhard riemann influence";

            Collection<String> strings = paragraphVectors.nearestLabels(queryString, 5);
            for (String s : strings) {
                Document document = reader.document(Integer.parseInt(s.substring(4)));
                System.out.println(document.get(fieldName));
            }

            try {
                IndexSearcher searcher = new IndexSearcher(reader);
                searcher.setSimilarity(new ParagraphVectorsSimilarity(paragraphVectors, fieldName));

                INDArray queryParagraphVector = paragraphVectors.inferVector(queryString);

                QueryParser parser = new QueryParser(fieldName, new WhitespaceAnalyzer());
                Query query = parser.parse(queryString);
                TopDocs hits = searcher.search(query, 10);
                for (int i = 0; i < hits.scoreDocs.length; i++) {
                    ScoreDoc scoreDoc = hits.scoreDocs[i];
                    Document doc = searcher.doc(scoreDoc.doc);

                    String title = doc.get(fieldName);
                    System.out.println(title + " : " + scoreDoc.score);

                    String label = "doc_" + scoreDoc.doc;
                    INDArray documentParagraphVector = paragraphVectors.getLookupTable().vector(label);
                    if (documentParagraphVector == null) {
                        LabelledDocument document = new LabelledDocument();
                        document.setLabels(Collections.singletonList(label));
                        document.setContent(title);
                        documentParagraphVector = paragraphVectors.inferVector(document);
                    }

                    System.out.println("cosineSimilarityParagraph=" + Transforms.cosineSim(queryParagraphVector, documentParagraphVector));
                }

            } finally {
                WordVectorSerializer.writeParagraphVectors(paragraphVectors, "target/ch5pv.zip");
                writer.deleteAll();
                writer.commit();
                writer.close();
                reader.close();
            }

        } finally {
            directory.close();
        }
    }
}
