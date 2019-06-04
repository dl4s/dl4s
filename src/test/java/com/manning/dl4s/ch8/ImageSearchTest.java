package com.manning.dl4s.ch8;

import com.google.common.io.Files;
import net.semanticmetadata.lire.aggregators.BOVW;
import net.semanticmetadata.lire.classifiers.Cluster;
import net.semanticmetadata.lire.classifiers.KMeans;
import net.semanticmetadata.lire.imageanalysis.features.LocalFeature;
import net.semanticmetadata.lire.imageanalysis.features.global.SimpleColorHistogram;
import net.semanticmetadata.lire.imageanalysis.features.local.sift.SiftExtractor;
import org.apache.commons.io.IOUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.document.*;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Tests for chapter 8
 */
public class ImageSearchTest {

  @Test
  public void testCifarWithPPCAReducedLeNet() throws Exception {

    int height = 32;
    int width = 32;
    int channels = 3;
    int numLabels = CifarLoader.NUM_LABELS;
    int numSamples = 10000;
    int batchSize = 100;
    int freIterations = 50;
    int seed = 123;
    boolean preProcessCifar = false;
    int epochs = 20;
    int features = 300;

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .list()
        .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}).name("cnn1").convolutionMode(ConvolutionMode.Same)
            .nIn(3).nOut(28).activation(Activation.RELU).build())
        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).name("maxpool1").build())

        .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {0,0}).name("cnn2").convolutionMode(ConvolutionMode.Same)
            .nOut(10).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
            .biasInit(1e-2).build())
        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}).name("maxpool2").build())
        .layer(4, new DenseLayer.Builder().name("ffn1").nOut(features).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .name("output")
            .nOut(numLabels)
            .activation(Activation.SOFTMAX)
            .build())
        .backprop(true)
        .pretrain(false)
        .setInputType(InputType.convolutional(height, width, channels))
        .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    model.setListeners(new ScoreIterationListener(freIterations));
    CifarDataSetIterator dsi = new CifarDataSetIterator(batchSize, numSamples, new int[] {height, width, channels},
            preProcessCifar, true);
    List<String> cifarLabels = dsi.getLabels();
    System.out.println(String.join(",", cifarLabels.toArray(new String[cifarLabels.size()])));

    for (int i = 0; i < epochs; ++i) {
      model.fit(dsi);
    }

    int rows = 1000;
    INDArray weights = Nd4j.zeros(rows, features);

    dsi.reset();
    int i = 0;
    List<String> stringLabels = new LinkedList<>();
    while (dsi.hasNext()) {
      DataSet batch = dsi.next(batchSize);
      for (int k = 0; k < batchSize; k++) {
        DataSet dataSet = batch.get(k);
        INDArray labels = dataSet.getLabels();
        stringLabels.add(cifarLabels.get(labels.argMax(1).maxNumber().intValue()));
        List<INDArray> activations = model.feedForward(dataSet.getFeatures(), false);
        INDArray imageRepresentation = activations.get(activations.size() - 2);
        float[] aFloat = imageRepresentation.data().asFloat();
        int idx = k + i;
        if (idx >= rows) {
          break;
        }
        weights.putRow(idx, Nd4j.create(aFloat));
      }

      i += batchSize;
      if (i >= rows) {
        break;
      }
    }

    int d = 8;
    INDArray x = postProcess(weights, d);
    INDArray pcaX = PCA.pca(x, d, true);
    INDArray reduced = postProcess(pcaX, d);

    // index all
    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);
    Map<String, Analyzer> mappings = new HashMap<>();
    mappings.put("lsh", new LSHAnalyzer());
    Analyzer perFieldAnalyzer = new PerFieldAnalyzerWrapper(new WhitespaceAnalyzer(), mappings);
    IndexWriterConfig config = new IndexWriterConfig(perFieldAnalyzer);

    IndexWriter writer = new IndexWriter(directory, config);
    try {
        int k = 0;
        for (String sl : stringLabels) {
          Document doc = new Document();
          float[] reducedPoint = reduced.getRow(k).toFloatVector();
          float[] fv = weights.getRow(k).toFloatVector();
          String fvString = toString(fv);
          doc.add(new FloatPoint("features", reducedPoint));
          doc.add(new TextField("label", sl, Field.Store.YES));
          doc.add(new TextField("lsh", fvString, Field.Store.YES));
          writer.addDocument(doc);
          k++;
        }
        writer.commit();

        // query by image
        DirectoryReader reader = DirectoryReader.open(writer);
        IndexSearcher searcher = new IndexSearcher(reader);

        Random r = new Random();
        System.out.println("comparing knn and lsh results");
        for (int counter = 0; counter < 10; counter++) {
            int docID = r.nextInt(reader.numDocs() - 1);
            Document document = reader.document(docID);
            System.out.println("image of a : " + document.get("label"));
            TopFieldDocs docs = FloatPointNearestNeighbor.nearest(searcher, "features", 3, reduced.getRow(docID).toFloatVector());
            ScoreDoc[] scoreDocs = docs.scoreDocs;
            for (ScoreDoc sd : scoreDocs) {
                System.out.println("knn : " + sd.doc +" : "+reader.document(sd.doc).getField("label").stringValue());
            }
            Query lshQuery = ImageQueryUtils.getSimilarityQuery(reader, docID, "lsh", 1);
            TopDocs topDocs = searcher.search(lshQuery, 3);
            for (ScoreDoc sd : topDocs.scoreDocs) {
              System.out.println("lsh : " + sd.doc +" : "+reader.document(sd.doc).getField("label").stringValue());
            }
            counter++;
        }

    } finally {
        writer.close();
        directory.close();
    }
  }

  private String toString(float[] a) {
      if (a == null)
        return "null";

      int iMax = a.length - 1;
      if (iMax == -1)
        return "";

      StringBuilder b = new StringBuilder();
      for (int i = 0; ; i++) {
        b.append(a[i]);
        if (i == iMax)
          return b.toString();
        b.append(" ");
      }
    }

  @Test
  public void testCifarWithVAE() throws Exception {

    //Neural net configuration
    int height = 32;
    int width = 32;
    int channels = 1;
    int numSamples = 2000;
    int batchSize = 128;
    boolean preProcessCifar = false;
    int epochs = 5;
    long rngSeed = 123;

    Nd4j.getRandom().setSeed(rngSeed);
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .updater(new RmsProp(1e-2))
            .weightInit(WeightInit.XAVIER)
            .l2(1e-4)
            .list()
            .layer(0, new VariationalAutoencoder.Builder()
                    .activation(Activation.SOFTSIGN)
                    .encoderLayerSizes(256, 128)
                    .decoderLayerSizes(256, 128)
                    .pzxActivationFunction(Activation.IDENTITY)
                    .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))
                    .nIn(height * width)
                    .nOut(8)
                    .build())
            .pretrain(true).backprop(false).build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    CifarDataSetIterator dsi = new CifarDataSetIterator(batchSize, numSamples, new int[] {height, width, channels},
            preProcessCifar, true);
    List<String> cifarLabels = dsi.getLabels();

    // reshaping cifar10 for VAE ingestion
    Collection<DataSet> reshapedData = new LinkedList<>();
    while (dsi.hasNext()) {
      DataSet batch = dsi.next(batchSize);
      for (int k = 0; k < batchSize; k++) {
        DataSet current = batch.get(k);
        DataSet dataSet = current.reshape(1, height * width);
        reshapedData.add(dataSet);
      }
    }
    dsi.reset();

    // pretrain VAE
    DataSetIterator trainingSet = new ListDataSetIterator<>(reshapedData);
    model.pretrain(trainingSet, epochs);

    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);
    IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());
    IndexWriter writer = new IndexWriter(directory, config);

    org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
            = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) model.getLayer(0);

    trainingSet.reset();
    List<float[]> featureList = new LinkedList<>();
    while (trainingSet.hasNext()) {
      DataSet batch = trainingSet.next(batchSize);
      for (int k = 0; k < batchSize; k++) {
        DataSet dataSet = batch.get(k);
        INDArray labels = dataSet.getLabels();
        String label = cifarLabels.get(labels.argMax(1).maxNumber().intValue());

        INDArray latentSpaceValues = vae.activate(dataSet.getFeatures(), false, LayerWorkspaceMgr.noWorkspaces());
        float[] aFloat = latentSpaceValues.data().asFloat();
        Document doc = new Document();
        doc.add(new FloatPoint("features", aFloat));
        doc.add(new TextField("label", label, Field.Store.YES));
        writer.addDocument(doc);
        featureList.add(aFloat);
      }
    }
    writer.commit();

    try {
      // query by image
      DirectoryReader reader = DirectoryReader.open(writer);
      IndexSearcher searcher = new IndexSearcher(reader);

      Random r = new Random();
      for (int counter = 0; counter < 10; counter++) {
        int idx = r.nextInt(reader.numDocs() - 1);
        Document document = reader.document(idx);
        TopFieldDocs docs = FloatPointNearestNeighbor.nearest(searcher, "features", 3, featureList.get(idx));
        ScoreDoc[] scoreDocs = docs.scoreDocs;
        System.out.println("query image of a : " + document.get("label"));
        for (ScoreDoc sd : scoreDocs) {
          System.out.println("-->" + sd.doc +" : "+reader.document(sd.doc).getField("label").stringValue());
        }
        counter++;
      }

    } finally {
      writer.close();
      directory.close();
    }
  }

  @Test
  public void testTBIR() throws Exception {

    Map<String, String> map = new HashMap<>();
    map.put("white sports car", "https://img.freepik.com/free-psd/car-mock-up-isolated-design_1310-1240.jpg?size=626&ext=jpg");
    map.put("black fluo sports car", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8LwFsFvQEltxdbOx707iwR5viA_kRV35Bb_Z_Kf7wdmcz4WbE");
    map.put("black sports car", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTzSFtP77bsTphjc92oP-pd8zC48GAKkfcysJsnVY5oH98rdODf");

    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);
    IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());
    IndexWriter writer = new IndexWriter(directory, config);

    try {

      int i = 0;
      for (Map.Entry<String, String> e : map.entrySet()) {
        String imgPath = "target/image" + i + ".png";
        Path target = Paths.get(imgPath);
        if (!target.toFile().exists()) {
          URL url = new URL(e.getValue());
          InputStream in = url.openStream();
          java.nio.file.Files.copy(in, target);
          in.close();
        }
        i++;

        Document doc = new Document();
        doc.add(new StoredField("binary", IOUtils.toByteArray(new FileInputStream(new File(imgPath)))));
        doc.add(new TextField("caption", e.getKey(), Field.Store.NO));
        writer.addDocument(doc);
      }
      writer.commit();

      DirectoryReader reader = DirectoryReader.open(writer);
      IndexSearcher searcher = new IndexSearcher(reader);

      TopDocs topDocs = searcher.search(new PhraseQuery("caption", "black", "sports", "car"), 3);
      assertEquals(1, topDocs.totalHits);
      for (ScoreDoc sd : topDocs.scoreDocs) {
        Document document = reader.document(sd.doc);
        IndexableField binary = document.getField("binary");
        BytesRef imageBinary = binary.binaryValue();
        assertNotNull(imageBinary);
      }

    } finally {
      writer.close();
      directory.close();
    }
  }

  @Test
  public void testHistogram() throws Exception {

    Map<String, String> map = new HashMap<>();
    map.put("white sports car", "https://img.freepik.com/free-psd/car-mock-up-isolated-design_1310-1240.jpg?size=626&ext=jpg");
    map.put("black fluo sports car", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8LwFsFvQEltxdbOx707iwR5viA_kRV35Bb_Z_Kf7wdmcz4WbE");
    map.put("black sports car", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTzSFtP77bsTphjc92oP-pd8zC48GAKkfcysJsnVY5oH98rdODf");

    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);
    IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());
    IndexWriter writer = new IndexWriter(directory, config);

    INDArray weights = null;
    try {

      int i = 0;
      for (Map.Entry<String, String> e : map.entrySet()) {
        String imgPath = "target/image" + i + ".png";
        Path target = Paths.get(imgPath);
        if (!target.toFile().exists()) {
          URL url = new URL(e.getValue());
          InputStream in = url.openStream();
          java.nio.file.Files.copy(in, target);
          in.close();
        }

        File file = new File(imgPath);
        SimpleColorHistogram simpleColorHistogram = new SimpleColorHistogram();
        BufferedImage bufferedImage = ImageIO.read(file);
        simpleColorHistogram.extract(bufferedImage);

        double[] featureVector = simpleColorHistogram.getFeatureVector();
        if (weights == null) {
          weights = Nd4j.zeros(map.size(), featureVector.length);
        }
        float[] floats = new float[featureVector.length];
        for (int j = 0; j < floats.length; j++) {
          floats[j] = (float) featureVector[j];
        }

        weights.putRow(i, Nd4j.create(floats));
        i++;
      }

      INDArray reduced = PCA.pca(weights, 8, true);

        // index
      int k = 0;
      for (Map.Entry<String, String> e : map.entrySet()) {
        Document doc = new Document();
        doc.add(new FloatPoint("features", reduced.getRow(k).toFloatVector()));
        doc.add(new TextField("caption", e.getKey(), Field.Store.YES));
        writer.addDocument(doc);
        k++;
      }
      writer.commit();

      DirectoryReader reader = DirectoryReader.open(writer);
      IndexSearcher searcher = new IndexSearcher(reader);

      int rowId = 0;
      while (rowId < weights.rows() ){
        Document document = reader.document(rowId);
        TopFieldDocs docs = FloatPointNearestNeighbor.nearest(searcher, "features", 2, reduced.getRow(rowId).toFloatVector());
        ScoreDoc[] scoreDocs = docs.scoreDocs;
        System.out.println(document.get("caption"));
        for (ScoreDoc sd : scoreDocs) {
          System.out.println("-->" + reader.document(sd.doc).getField("caption"));
        }
        rowId++;
      }

    } finally {
      writer.close();
      directory.close();
    }
  }

  private INDArray postProcess(INDArray weights, int d) {
      INDArray meanWeights = weights.sub(weights.meanNumber());
      INDArray pca = PCA.pca(meanWeights, d, true);
      for (int j = 0; j < weights.rows(); j++) {
          INDArray v = meanWeights.getRow(j);
          for (int s = 0; s < d; s++) {
              INDArray u = pca.getColumn(s);
              INDArray mul = u.mmul(v).transpose().mmul(u);
              v.subi(mul.transpose());
          }
      }
      return weights;
  }

  @Test
  public void testSIFTBoVW() throws Exception {

    Map<String, String> map = new HashMap<>();
    map.put("white sports car", "https://img.freepik.com/free-psd/car-mock-up-isolated-design_1310-1240.jpg?size=626&ext=jpg");
    map.put("black fluo sports car", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8LwFsFvQEltxdbOx707iwR5viA_kRV35Bb_Z_Kf7wdmcz4WbE");
    map.put("black sports car", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTzSFtP77bsTphjc92oP-pd8zC48GAKkfcysJsnVY5oH98rdODf");

    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);
    IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());
    IndexWriter writer = new IndexWriter(directory, config);

    KMeans kMeans = new KMeans(8);

    Map<String, float[]> features = new HashMap<>();
    try {

      int i = 0;
      for (Map.Entry<String, String> e : map.entrySet()) {
        String imgPath = "target/image" + i + ".png";
        Path target = Paths.get(imgPath);
        if (!target.toFile().exists()) {
          URL url = new URL(e.getValue());
          InputStream in = url.openStream();
          java.nio.file.Files.copy(in, target);
          in.close();
        }
        i++;

        File file = new File(imgPath);
        SiftExtractor siftExtractor = new SiftExtractor();
        BufferedImage bufferedImage = ImageIO.read(file);
        siftExtractor.extract(bufferedImage);
        List<? extends LocalFeature> localFeatures = siftExtractor.getFeatures();
        for (LocalFeature lf : localFeatures) {
          kMeans.addFeature(lf.getFeatureVector());
        }

      }
      kMeans.init();
      for (int k = 0; k < 15; k++) {
        kMeans.clusteringStep();
      }
      Cluster[] clusters = kMeans.getClusters();

      i = 0;
      for (Map.Entry<String, String> e : map.entrySet()) {
        String imgPath = "target/image" + i + ".png";
        i++;

        Document doc = new Document();
        File file = new File(imgPath);
        SiftExtractor siftExtractor = new SiftExtractor();
        BufferedImage bufferedImage = ImageIO.read(file);
        siftExtractor.extract(bufferedImage);
        List<? extends LocalFeature> localFeatures = siftExtractor.getFeatures();
        BOVW bovw = new BOVW();
        bovw.createVectorRepresentation(localFeatures, clusters);
        double[] featureVector = bovw.getVectorRepresentation();
        float[] floats = new float[featureVector.length];
        for (int j = 0; j < floats.length; j++) {
          floats[j] = (float) featureVector[j];
        }
        features.put(e.getKey(), floats);
        doc.add(new FloatPoint("features", floats));
        doc.add(new StoredField("binary", IOUtils.toByteArray(new FileInputStream(file))));
        doc.add(new TextField("caption", e.getKey(), Field.Store.YES));
        writer.addDocument(doc);
      }
      writer.commit();

      DirectoryReader reader = DirectoryReader.open(writer);
      IndexSearcher searcher = new IndexSearcher(reader);

      for (Map.Entry<String, float[]> entry : features.entrySet()) {
        TopFieldDocs docs = FloatPointNearestNeighbor.nearest(searcher, "features", 2, entry.getValue());
        ScoreDoc[] scoreDocs = docs.scoreDocs;
        System.out.println(entry.getKey());
        for (ScoreDoc sd : scoreDocs) {
          System.out.println("-->" + reader.document(sd.doc).getField("caption"));
        }
      }

    } finally {
      writer.close();
      directory.close();
    }
  }

}
