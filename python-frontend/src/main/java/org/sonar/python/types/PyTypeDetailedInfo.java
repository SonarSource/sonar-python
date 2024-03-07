package org.sonar.python.types;

public class PyTypeDetailedInfo {
  private final String raw;

  public PyTypeDetailedInfo(String raw) {
    this.raw = raw;
  }

  public String raw() {
    return raw;
  }
}
