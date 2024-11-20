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
package org.sonar.python.metrics;

import java.io.File;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;

import static org.fest.assertions.Assertions.assertThat;

class CognitiveComplexityVisitorTest {

  @Test
  void file() {
    Map<Integer, String> complexityByLine = new TreeMap<>();
    CognitiveComplexityVisitor fileComplexityVisitor = new CognitiveComplexityVisitor((token, message) -> complexityByLine.merge(token.line(), message, (a, b) -> a + " " + b));


    BaseTreeVisitor functionVisitor = new BaseTreeVisitor() {
      @Override
      public void visitFunctionDef(FunctionDef tree) {
        int functionComplexity = CognitiveComplexityVisitor.complexity(tree, null);
        complexityByLine.merge(tree.firstToken().line(), "=" + functionComplexity, (a, b) -> a + " " + b);
      }
    };

    PythonVisitorContext context = TestPythonVisitorRunner.createContext(new File("src/test/resources/metrics/cognitive-complexities.py"));
    context.rootTree().accept(fileComplexityVisitor);
    context.rootTree().accept(functionVisitor);
    CommentVisitor commentVisitor = new CommentVisitor();
    commentVisitor.scanFile(context);
    assertThat(fileComplexityVisitor.getComplexity()).isEqualTo(91);

    StringBuilder complexityReport = new StringBuilder();
    complexityByLine.forEach((line, message) -> complexityReport.append("line " + line + " " + message + "\n"));
    assertThat(complexityReport.toString()).isEqualTo(commentVisitor.comments.toString());
  }

  private static class CommentVisitor extends PythonSubscriptionCheck {
    StringBuilder comments = new StringBuilder();

    @Override
    public void scanFile(PythonVisitorContext visitorContext) {
      SubscriptionVisitor.analyze(Collections.singletonList(this), visitorContext);
    }

    @Override
    public void initialize(Context context) {
      context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
        for (Trivia trivia : ((Token) ctx.syntaxNode()).trivia()) {
          String content = trivia.token().value().substring(1).trim();
          if (content.startsWith("=") || content.startsWith("+")) {
            comments.append("line ")
              .append(trivia.token().line())
              .append(" ")
              .append(content)
              .append("\n");
          }
        }
      });
    }
  }

}
