/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

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

  public static IssueLocation preciseLocation(AstNode startNode, AstNode endNode, String message) {
    return new PreciseIssueLocation(startNode, endNode, message);
  }

  public static IssueLocation preciseLocation(AstNode startNode, String message) {
    return new PreciseIssueLocation(startNode, message);
  }

  @CheckForNull
  public String message() {
    return message;
  }

  public abstract int startLine();

  public abstract int startLineOffset();

  public abstract int endLine();

  public abstract int endLineOffset();

  private static class PreciseIssueLocation extends IssueLocation {

    private final Token firstToken;
    private final TokenLocation lastTokenLocation;

    public PreciseIssueLocation(AstNode node, String message) {
      super(message);
      this.firstToken = node.getToken();
      this.lastTokenLocation = new TokenLocation(node.getLastToken());
    }

    public PreciseIssueLocation(AstNode startNode, AstNode endNode, String message) {
      super(message);
      this.firstToken = startNode.getToken();
      this.lastTokenLocation = new TokenLocation(endNode.getLastToken());
    }

    @Override
    public int startLine() {
      return firstToken.getLine();
    }

    @Override
    public int startLineOffset() {
      return firstToken.getColumn();
    }

    @Override
    public int endLine() {
      return lastTokenLocation.endLine();
    }

    @Override
    public int endLineOffset() {
      return lastTokenLocation.endLineOffset();
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

  }
}
