/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.api.quickfix;

import java.util.Objects;
import org.sonar.api.Beta;

@Beta
public class PythonTextEdit {

  private final String message;
  private final int startLine;
  private final int startLineOffset;
  private final int endLine;
  private final int endLineOffset;

  public PythonTextEdit(String message, int startLine, int startLineOffset, int endLine, int endLineOffset) {
    this.message = message;
    this.startLine = startLine;
    this.startLineOffset = startLineOffset;
    this.endLine = endLine;
    this.endLineOffset = endLineOffset;
  }

  public String replacementText() {
    return message;
  }

  public int startLine() {
    return startLine;
  }

  public int startLineOffset() {
    return startLineOffset;
  }

  public int endLine() {
    return endLine;
  }

  public int endLineOffset() {
    return endLineOffset;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    PythonTextEdit that = (PythonTextEdit) o;
    return startLine == that.startLine && startLineOffset == that.startLineOffset && endLine == that.endLine
      && endLineOffset == that.endLineOffset && Objects.equals(message, that.message);
  }

  @Override
  public int hashCode() {
    return Objects.hash(message, startLine, startLineOffset, endLine, endLineOffset);
  }

  @Override
  public String toString() {
    return "PythonTextEdit{" +
      "message='" + message + '\'' +
      ", startLine=" + startLine +
      ", startLineOffset=" + startLineOffset +
      ", endLine=" + endLine +
      ", endLineOffset=" + endLineOffset +
      '}';
  }
}
