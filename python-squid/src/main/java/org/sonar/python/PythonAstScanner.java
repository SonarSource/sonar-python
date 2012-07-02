/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import com.sonar.sslr.api.CommentAnalyser;
import com.sonar.sslr.impl.Parser;
import com.sonar.sslr.squid.*;
import com.sonar.sslr.squid.metrics.CommentsVisitor;
import com.sonar.sslr.squid.metrics.ComplexityVisitor;
import com.sonar.sslr.squid.metrics.CounterVisitor;
import com.sonar.sslr.squid.metrics.LinesVisitor;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonMetric;
import org.sonar.python.parser.PythonParser;
import org.sonar.squid.api.SourceCode;
import org.sonar.squid.api.SourceFile;
import org.sonar.squid.api.SourceFunction;
import org.sonar.squid.api.SourceProject;
import org.sonar.squid.indexer.QueryByType;

import java.io.File;
import java.util.Collection;

public final class PythonAstScanner {

  private PythonAstScanner() {
  }

  /**
   * Helper method for testing checks without having to deploy them on a Sonar instance.
   */
  public static SourceFile scanSingleFile(File file, SquidAstVisitor<PythonGrammar>... visitors) {
    if (!file.isFile()) {
      throw new IllegalArgumentException("File '" + file + "' not found.");
    }
    AstScanner<PythonGrammar> scanner = create(new PythonConfiguration(), visitors);
    scanner.scanFile(file);
    Collection<SourceCode> sources = scanner.getIndex().search(new QueryByType(SourceFile.class));
    if (sources.size() != 1) {
      throw new IllegalStateException("Only one SourceFile was expected whereas " + sources.size() + " has been returned.");
    }
    return (SourceFile) sources.iterator().next();
  }

  public static AstScanner<PythonGrammar> create(PythonConfiguration conf, SquidAstVisitor<PythonGrammar>... visitors) {
    final SquidAstVisitorContextImpl<PythonGrammar> context = new SquidAstVisitorContextImpl<PythonGrammar>(new SourceProject("Python Project"));
    final Parser<PythonGrammar> parser = PythonParser.create(conf);

    AstScanner.Builder<PythonGrammar> builder = AstScanner.<PythonGrammar> builder(context).setBaseParser(parser);

    /* Metrics */
    builder.withMetrics(PythonMetric.values());

    /* Files */
    builder.setFilesMetric(PythonMetric.FILES);

    /* Comments */
    builder.setCommentAnalyser(
        new CommentAnalyser() {
          @Override
          public boolean isBlank(String line) {
            for (int i = 0; i < line.length(); i++) {
              if (Character.isLetterOrDigit(line.charAt(i))) {
                return false;
              }
            }
            return true;
          }

          @Override
          public String getContents(String comment) {
            // Comment always starts with "#"
            return comment.substring(comment.indexOf('#'));
          }
        });

    /* Functions */
    builder.withSquidAstVisitor(new SourceCodeBuilderVisitor<PythonGrammar>(new SourceCodeBuilderCallback() {
      public SourceCode createSourceCode(SourceCode parentSourceCode, AstNode astNode) {
        String functionName = astNode.findFirstChild(parser.getGrammar().funcname).getChild(0).getTokenValue();
        SourceFunction function = new SourceFunction(functionName + ":" + astNode.getToken().getLine());
        function.setStartAtLine(astNode.getTokenLine());
        return function;
      }
    }, parser.getGrammar().funcdef));

    builder.withSquidAstVisitor(CounterVisitor.<PythonGrammar> builder()
        .setMetricDef(PythonMetric.FUNCTIONS)
        .subscribeTo(parser.getGrammar().funcdef)
        .build());

    /* Classes */
    builder.withSquidAstVisitor(new SourceCodeBuilderVisitor<PythonGrammar>(new SourceCodeBuilderCallback() {
      public SourceCode createSourceCode(SourceCode parentSourceCode, AstNode astNode) {
        String functionName = astNode.findFirstChild(parser.getGrammar().classname).getChild(0).getTokenValue();
        SourceFunction function = new SourceFunction(functionName + ":" + astNode.getToken().getLine());
        function.setStartAtLine(astNode.getTokenLine());
        return function;
      }
    }, parser.getGrammar().classdef));

    builder.withSquidAstVisitor(CounterVisitor.<PythonGrammar> builder()
        .setMetricDef(PythonMetric.CLASSES)
        .subscribeTo(parser.getGrammar().classdef)
        .build());

    /* Metrics */
    builder.withSquidAstVisitor(new LinesVisitor<PythonGrammar>(PythonMetric.LINES));
    builder.withSquidAstVisitor(new PythonLinesOfCodeVisitor<PythonGrammar>(PythonMetric.LINES_OF_CODE));
    builder.withSquidAstVisitor(CommentsVisitor.<PythonGrammar> builder().withCommentMetric(PythonMetric.COMMENT_LINES)
        .withBlankCommentMetric(PythonMetric.COMMENT_BLANK_LINES)
        .withNoSonar(true)
        .withIgnoreHeaderComment(conf.getIgnoreHeaderComments())
        .build());
    builder.withSquidAstVisitor(CounterVisitor.<PythonGrammar> builder()
        .setMetricDef(PythonMetric.STATEMENTS)
        .subscribeTo(parser.getGrammar().statement)
        .build());

    AstNodeType[] complexityAstNodeType = new AstNodeType[] {
      // Entry points
      parser.getGrammar().funcdef,

      // Branching nodes
      parser.getGrammar().if_stmt,
      parser.getGrammar().while_stmt,
      parser.getGrammar().for_stmt,
      parser.getGrammar().return_stmt,
      parser.getGrammar().raise_stmt,
        // TODO add catch
        // TODO Expressions: TODO ?, &&, ||
    };
    builder.withSquidAstVisitor(ComplexityVisitor.<PythonGrammar> builder()
        .setMetricDef(PythonMetric.COMPLEXITY)
        .subscribeTo(complexityAstNodeType)
        .build());

    /* External visitors (typically Check ones) */
    for (SquidAstVisitor<PythonGrammar> visitor : visitors) {
      builder.withSquidAstVisitor(visitor);
    }

    return builder.build();
  }

}
