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
package org.sonar.python;

import com.google.common.base.Charsets;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.impl.Parser;
import java.io.File;
import java.util.Collection;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonMetric;
import org.sonar.python.parser.PythonParser;
import org.sonar.squidbridge.AstScanner;
import org.sonar.squidbridge.CommentAnalyser;
import org.sonar.squidbridge.SourceCodeBuilderCallback;
import org.sonar.squidbridge.SourceCodeBuilderVisitor;
import org.sonar.squidbridge.SquidAstVisitor;
import org.sonar.squidbridge.SquidAstVisitorContextImpl;
import org.sonar.squidbridge.api.SourceClass;
import org.sonar.squidbridge.api.SourceCode;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.api.SourceFunction;
import org.sonar.squidbridge.api.SourceProject;
import org.sonar.squidbridge.indexer.QueryByType;
import org.sonar.squidbridge.metrics.ComplexityVisitor;
import org.sonar.squidbridge.metrics.CounterVisitor;
import org.sonar.squidbridge.metrics.LinesVisitor;

public final class PythonAstScanner {

  private PythonAstScanner() {
  }

  /**
   * Helper method for testing checks without having to deploy them on a Sonar instance.
   */
  @SuppressWarnings("unchecked")
  public static SourceFile scanSingleFile(String path, SquidAstVisitor<Grammar>... visitors) {
    File file = new File(path);
    if (!file.isFile()) {
      throw new IllegalArgumentException("File '" + file + "' not found.");
    }
    AstScanner<Grammar> scanner = create(new PythonConfiguration(Charsets.UTF_8), visitors);
    scanner.scanFile(file);
    Collection<SourceCode> sources = scanner.getIndex().search(new QueryByType(SourceFile.class));
    if (sources.size() != 1) {
      throw new IllegalStateException("Only one SourceFile was expected whereas " + sources.size() + " has been returned.");
    }
    return (SourceFile) sources.iterator().next();
  }

  public static AstScanner<Grammar> create(PythonConfiguration conf, SquidAstVisitor<Grammar>... visitors) {
    final SquidAstVisitorContextImpl<Grammar> context = new SquidAstVisitorContextImpl<>(new SourceProject("Python Project"));
    final Parser<Grammar> parser = PythonParser.create(conf);

    AstScanner.Builder<Grammar> builder = AstScanner.<Grammar>builder(context).setBaseParser(parser);

    builder.withMetrics(PythonMetric.values());

    builder.setFilesMetric(PythonMetric.FILES);

    setCommentAnalyser(builder);

    setClassesAnalyser(builder);

    setMethodAnalyser(builder);

    setMetrics(builder);

    /* External visitors (typically Check ones) */
    for (SquidAstVisitor<Grammar> visitor : visitors) {
      if (visitor instanceof CharsetAwareVisitor) {
        ((CharsetAwareVisitor) visitor).setCharset(conf.getCharset());
      }
      builder.withSquidAstVisitor(visitor);
    }

    return builder.build();
  }

  private static void setMetrics(AstScanner.Builder<Grammar> builder) {
    builder.withSquidAstVisitor(new LinesVisitor<Grammar>(PythonMetric.LINES));
    AstNodeType[] complexityAstNodeType = new AstNodeType[]{
      // Entry points
      PythonGrammar.FUNCDEF,

      // Branching nodes
      // Note that IF_STMT covered by PythonKeyword.IF below
      PythonGrammar.WHILE_STMT,
      PythonGrammar.FOR_STMT,
      PythonGrammar.RETURN_STMT,
      PythonGrammar.RAISE_STMT,
      PythonGrammar.EXCEPT_CLAUSE,

      // Expressions
      PythonKeyword.IF,
      PythonKeyword.AND,
      PythonKeyword.OR
    };

    builder.withSquidAstVisitor(ComplexityVisitor.<Grammar>builder()
      .setMetricDef(PythonMetric.COMPLEXITY)
      .subscribeTo(complexityAstNodeType)
      .build());

    builder.withSquidAstVisitor(CounterVisitor.<Grammar>builder()
      .setMetricDef(PythonMetric.STATEMENTS)
      .subscribeTo(PythonGrammar.STATEMENT)
      .build());
  }

  private static void setMethodAnalyser(AstScanner.Builder<Grammar> builder) {
    SourceCodeBuilderCallback sourceCodeBuilderCallback = (SourceCode parentSourceCode, AstNode astNode) -> {
      String functionName = astNode.getFirstChild(PythonGrammar.FUNCNAME).getFirstChild().getTokenValue();
      SourceFunction function = new SourceFunction(functionName + ":" + astNode.getToken().getLine());
      function.setStartAtLine(astNode.getTokenLine());
      return function;
    };

    builder.withSquidAstVisitor(new SourceCodeBuilderVisitor<>(sourceCodeBuilderCallback, PythonGrammar.FUNCDEF));

    builder.withSquidAstVisitor(CounterVisitor.<Grammar>builder()
      .setMetricDef(PythonMetric.FUNCTIONS)
      .subscribeTo(PythonGrammar.FUNCDEF)
      .build());
  }

  private static void setClassesAnalyser(AstScanner.Builder<Grammar> builder) {
    SourceCodeBuilderCallback sourceCodeBuilderCallback = (SourceCode parentSourceCode, AstNode astNode) -> {
      String functionName = astNode.getFirstChild(PythonGrammar.CLASSNAME).getFirstChild().getTokenValue();
      SourceClass function = new SourceClass(functionName + ":" + astNode.getToken().getLine());
      function.setStartAtLine(astNode.getTokenLine());
      return function;
    };

    builder.withSquidAstVisitor(new SourceCodeBuilderVisitor<>(sourceCodeBuilderCallback, PythonGrammar.CLASSDEF));

    builder.withSquidAstVisitor(CounterVisitor.<Grammar>builder()
      .setMetricDef(PythonMetric.CLASSES)
      .subscribeTo(PythonGrammar.CLASSDEF)
      .build());
  }

  private static void setCommentAnalyser(AstScanner.Builder<Grammar> builder) {
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
  }

}
