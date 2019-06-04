package com.manning.dl4s.ch9;

import java.io.Serializable;
import java.util.Objects;

public class Tweet implements Serializable {

  private String id;
  private String text;
  private String language;
  private String user;

  public Tweet(String id, String text, String language, String user) {
    this.id = id;
    this.text = text;
    this.language = language;
    this.user = user;
  }

  public String getId() {
    return id;
  }

  public void setId(String id) {
    this.id = id;
  }

  public String getText() {
    return text;
  }

  public void setText(String text) {
    this.text = text;
  }

  public String getLanguage() {
    return language;
  }

  public void setLanguage(String language) {
    this.language = language;
  }

  public String getUser() {
    return user;
  }

  public void setUser(String user) {
    this.user = user;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Tweet tweet = (Tweet) o;
    return Objects.equals(getId(), tweet.getId()) &&
        Objects.equals(getText(), tweet.getText()) &&
        Objects.equals(getLanguage(), tweet.getLanguage()) &&
        Objects.equals(getUser(), tweet.getUser());
  }

  @Override
  public int hashCode() {
    return Objects.hash(getId(), getText(), getLanguage(), getUser());
  }

  @Override
  public String toString() {
    return text.trim();
  }
}
