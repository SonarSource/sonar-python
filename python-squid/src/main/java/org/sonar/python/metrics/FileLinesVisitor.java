/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.python.metrics;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.sonar.sslr.api.AstAndTokenVisitor;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import java.util.Map;
import java.util.Set;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.python.DocstringExtractor;
import org.sonar.python.TokenLocation;
import org.sonar.python.api.PythonMetric;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.SquidAstVisitor;
import org.sonar.squidbridge.api.SourceFile;

/**
 * Visitor that computes {@link CoreMetrics#NCLOC_DATA_KEY} and {@link CoreMetrics#COMMENT_LINES_DATA_KEY} metrics used by the DevCockpit.
 */
public class FileLinesVisitor extends SquidAstVisitor<Grammar> implements AstAndTokenVisitor {

  private final FileLinesContextFactory fileLinesContextFactory;

  private boolean seenFirstToken;

  private final boolean ignoreHeaderComments;

  private Set<Integer> noSonar = Sets.newHashSet();
  private Set<Integer> linesOfCode = Sets.newHashSet();
  private Set<Integer> linesOfComments = Sets.newHashSet();
  private Set<Integer> linesOfDocstring = Sets.newHashSet();
  private final FileSystem fileSystem;
  private final Map<InputFile, Set<Integer>> allLinesOfCode;

  public FileLinesVisitor(
      FileLinesContextFactory fileLinesContextFactory,
      FileSystem fileSystem,
      Map<InputFile, Set<Integer>> linesOfCode,
      boolean ignoreHeaderComments) {
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.fileSystem = fileSystem;
    this.allLinesOfCode = linesOfCode;
    this.ignoreHeaderComments = ignoreHeaderComments;
  }

  @Override
  public void init() {
    DocstringExtractor.DOCUMENTABLE_NODE_TYPES.stream().forEach(this::subscribeTo);
  }

  @Override
  public void visitFile(AstNode astNode) {
    noSonar.clear();
    linesOfCode.clear();
    linesOfComments.clear();
    linesOfDocstring.clear();
    seenFirstToken = false;
  }

  @Override
  public void visitNode(AstNode astNode) {
    Token docstringToken = DocstringExtractor.extractDocstring(astNode);
    if (docstringToken != null) {
      TokenLocation location = new TokenLocation(docstringToken);
      for (int line = location.startLine(); line <= location.endLine(); line++) {
        linesOfDocstring.add(line);
      }
    }
  }

  /**
   * Gets the lines of codes and lines of comments (with character #).
   * Does not get the lines of docstrings.
   */
  @Override
  public void visitToken(Token token) {
    if (token.getType().equals(GenericTokenType.EOF)) {
      return;
    }

    if (!token.getType().equals(PythonTokenType.DEDENT) && !token.getType().equals(PythonTokenType.INDENT) && !token.getType().equals(PythonTokenType.NEWLINE)) {
      //  Handle all the lines of the token
      String[] tokenLines = token.getValue().split("\n", -1);
      for (int line = token.getLine(); line < token.getLine() + tokenLines.length; line++) {
        linesOfCode.add(line);
      }
    }

    if (ignoreHeaderComments && !seenFirstToken) {
      seenFirstToken = true;
      return;
    }

    for (Trivia trivia : token.getTrivia()) {
      if (trivia.isComment()) {
        visitComment(trivia);
      }
    }
  }

  public void visitComment(Trivia trivia) {
    String[] commentLines = getContext().getCommentAnalyser().getContents(trivia.getToken().getOriginalValue())
      .split("(\r)?\n|\r", -1);
    int line = trivia.getToken().getLine();

    for (String commentLine : commentLines) {
      if (commentLine.contains("NOSONAR")) {
        linesOfComments.remove(line);
        noSonar.add(line);
      } else if (!getContext().getCommentAnalyser().isBlank(commentLine) && !noSonar.contains(line)) {
        linesOfComments.add(line);
      }
      line++;
    }
  }

  @Override
  public void leaveFile(AstNode astNode) {
    InputFile inputFile = fileSystem.inputFile(fileSystem.predicates().is(getContext().getFile()));
    if (inputFile == null){
      throw new IllegalStateException("InputFile is null, but it should not be.");
    }
    FileLinesContext fileLinesContext = fileLinesContextFactory.createFor(inputFile);

    // account for the docstring lines
    for (Integer line : linesOfDocstring) {
      linesOfCode.remove(line);
      linesOfComments.add(line);
    }

    int fileLength = getContext().peekSourceCode().getInt(PythonMetric.LINES);
    for (int line = 1; line <= fileLength; line++) {
      if (linesOfCode.contains(line)) {
        fileLinesContext.setIntValue(CoreMetrics.NCLOC_DATA_KEY, line, 1);
      }
      if (linesOfComments.contains(line)) {
        fileLinesContext.setIntValue(CoreMetrics.COMMENT_LINES_DATA_KEY, line, 1);
      }
    }
    fileLinesContext.save();

    allLinesOfCode.put(inputFile, ImmutableSet.copyOf(linesOfCode));

    getContext().peekSourceCode().add(PythonMetric.LINES_OF_CODE, linesOfCode.size());
    getContext().peekSourceCode().add(PythonMetric.COMMENT_LINES, linesOfComments.size());

    ((SourceFile) getContext().peekSourceCode()).addNoSonarTagLines(noSonar);
  }

}
