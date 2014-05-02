/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.metrics;

import com.google.common.collect.Sets;
import com.sonar.sslr.api.*;
import org.sonar.squidbridge.SquidAstVisitor;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.resources.File;
import org.sonar.api.resources.Project;
import org.sonar.python.api.PythonMetric;
import org.sonar.python.api.PythonTokenType;

import java.util.List;
import java.util.Set;

/**
 * Visitor that computes {@link CoreMetrics#NCLOC_DATA_KEY} and {@link CoreMetrics#COMMENT_LINES_DATA_KEY} metrics used by the DevCockpit.
 */
public class FileLinesVisitor extends SquidAstVisitor<Grammar> implements AstAndTokenVisitor {

  private final Project project;
  private final FileLinesContextFactory fileLinesContextFactory;

  private final Set<Integer> linesOfCode = Sets.newHashSet();
  private final Set<Integer> linesOfComments = Sets.newHashSet();

  public FileLinesVisitor(Project project, FileLinesContextFactory fileLinesContextFactory) {
    this.project = project;
    this.fileLinesContextFactory = fileLinesContextFactory;
  }

  public void visitToken(Token token) {
    if (token.getType().equals(GenericTokenType.EOF)) {
      return;
    }

    if (token.getType() != PythonTokenType.DEDENT && token.getType() != PythonTokenType.INDENT && token.getType() != PythonTokenType.NEWLINE) {
      /* Handle all the lines of the token */
      String[] tokenLines = token.getValue().split("\n", -1);
      for (int line = token.getLine(); line < token.getLine() + tokenLines.length; line++) {
        linesOfCode.add(line);
      }
    }

    List<Trivia> trivias = token.getTrivia();
    for (Trivia trivia : trivias) {
      if (trivia.isComment()) {
        linesOfComments.add(trivia.getToken().getLine());
      }
    }
  }

  @Override
  public void leaveFile(AstNode astNode) {
    File sonarFile = File.fromIOFile(getContext().getFile(), project);
    FileLinesContext fileLinesContext = fileLinesContextFactory.createFor(sonarFile);

    int fileLength = getContext().peekSourceCode().getInt(PythonMetric.LINES);
    for (int line = 1; line <= fileLength; line++) {
      fileLinesContext.setIntValue(CoreMetrics.NCLOC_DATA_KEY, line, linesOfCode.contains(line) ? 1 : 0);
      fileLinesContext.setIntValue(CoreMetrics.COMMENT_LINES_DATA_KEY, line, linesOfComments.contains(line) ? 1 : 0);
    }
    fileLinesContext.save();

    linesOfCode.clear();
    linesOfComments.clear();
  }

}
