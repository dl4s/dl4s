package com.manning.dl4s.ch9.functions.index;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Path;

import com.manning.dl4s.ch9.functions.Utils;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;

/**
 * Simple {@link IndexWriter} extension
 */
public class CustomWriter extends IndexWriter implements Serializable {
  private Directory directory;
  private IndexWriterConfig iwConfig;

  public CustomWriter() throws IOException {
    this(FSDirectory.open(Utils.getIndexPath(), NoLockFactory.INSTANCE), new IndexWriterConfig());
  }

  private CustomWriter(Directory dir, IndexWriterConfig iwConf) throws IOException {
    super(dir, iwConf);
  }

  public CustomWriter(Path path) throws IOException {
    this(FSDirectory.open(path), new IndexWriterConfig());
  }

  public Directory getDirectory() {
    return directory;
  }

  public IndexWriterConfig getConfig() {
    return iwConfig;
  }

  public void setDirectory(Directory dir) {
    directory = dir;
  }

  public void setConfig(IndexWriterConfig conf) {
    iwConfig = conf;
  }

  public Path getPath() {
    return Utils.getIndexPath();
  }
}