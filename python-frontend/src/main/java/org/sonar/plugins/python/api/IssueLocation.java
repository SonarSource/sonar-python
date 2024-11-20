/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python.api;

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.TokenLocation;

public abstract class IssueLocation {

  public static final int UNDEFINED_OFFSET = -1;

  public static final int UNDEFINED_LINE = 0;

  private String message;

  private IssueLocation(@Nullable String message) {
    this.message = message;
  }

  public static IssueLocation atFileLevel(String message) {
    return new FileLevelIssueLocation(message);
  }

  public static IssueLocation atLineLevel(String message, int lineNumber) {
    return new LineLevelIssueLocation(message, lineNumber);
  }

  public static IssueLocation preciseLocation(Tree tree, @Nullable String message) {
    return new PreciseIssueLocation(tree.firstToken(), tree.lastToken(), message);
  }

  public static IssueLocation preciseLocation(Token token, @Nullable String message) {
    return new PreciseIssueLocation(token, message);
  }

  public static IssueLocation preciseLocation(Token from, Token to, @Nullable String message) {
    return new PreciseIssueLocation(from, to, message);
  }

  public static IssueLocation preciseLocation(LocationInFile locationInFile, @Nullable String message) {
    return new PreciseIssueLocation(locationInFile, message);
  }

  @CheckForNull
  public String message() {
    return message;
  }

  public abstract int startLine();

  public abstract int startLineOffset();

  public abstract int endLine();

  public abstract int endLineOffset();

  @CheckForNull
  public abstract String fileId();

  private static class PreciseIssueLocation extends IssueLocation {
    @CheckForNull
    private final String fileId;
    private final int startLine;
    private final int startLineOffset;
    private final int endLine;
    private final int endLineOffset;

    public PreciseIssueLocation(Token firstToken, Token lastToken, @Nullable String message) {
      super(message);
      startLine = firstToken.line();
      startLineOffset = firstToken.column();
      TokenLocation tokenLocation = new TokenLocation(lastToken);
      endLine = tokenLocation.endLine();
      endLineOffset = tokenLocation.endLineOffset();
      fileId = null;
    }

    public PreciseIssueLocation(Token token, @Nullable String message) {
      this(token, token, message);
    }

    public PreciseIssueLocation(LocationInFile locationInFile, @Nullable String message) {
      super(message);
      startLine = locationInFile.startLine();
      startLineOffset = locationInFile.startLineOffset();
      endLine = locationInFile.endLine();
      endLineOffset = locationInFile.endLineOffset();
      fileId = locationInFile.fileId();
    }

    @Override
    public int startLine() {
      return startLine;
    }

    @Override
    public int startLineOffset() {
      return startLineOffset;
    }

    @Override
    public int endLine() {
      return endLine;
    }

    @Override
    public int endLineOffset() {
      return endLineOffset;
    }

    @Override
    public String fileId() {
      return fileId;
    }

  }

  private static class LineLevelIssueLocation extends IssueLocation {

    private final int lineNumber;

    public LineLevelIssueLocation(String message, int lineNumber) {
      super(message);
      this.lineNumber = lineNumber;
    }

    @Override
    public int startLine() {
      return lineNumber;
    }

    @Override
    public int startLineOffset() {
      return UNDEFINED_OFFSET;
    }

    @Override
    public int endLine() {
      return lineNumber;
    }

    @Override
    public int endLineOffset() {
      return UNDEFINED_OFFSET;
    }

    @Override
    public String fileId() {
      return null;
    }

  }

  private static class FileLevelIssueLocation extends IssueLocation {

    public FileLevelIssueLocation(String message) {
      super(message);
    }

    @Override
    public int startLine() {
      return UNDEFINED_LINE;
    }

    @Override
    public int startLineOffset() {
      return UNDEFINED_OFFSET;
    }

    @Override
    public int endLine() {
      return UNDEFINED_LINE;
    }

    @Override
    public int endLineOffset() {
      return UNDEFINED_OFFSET;
    }

    @Override
    public String fileId() {
      return null;
    }

  }
}
