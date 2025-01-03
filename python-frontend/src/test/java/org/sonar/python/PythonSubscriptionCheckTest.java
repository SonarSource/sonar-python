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
package org.sonar.python;

import java.io.File;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;

import static org.assertj.core.api.Assertions.assertThat;

class PythonSubscriptionCheckTest {

  private static final File FILE = new File("src/test/resources/file.py");
  public static final String MESSAGE = "message";

  private static List<PreciseIssue> scanFileForIssues(File file, PythonCheck check) {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(file);
    check.scanFile(context);
    SubscriptionVisitor.analyze(Collections.singletonList((PythonSubscriptionCheck) check), context);
    return context.getIssues();
  }

  @Test
  void test() {
    TestPythonCheck check = new TestPythonCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
          FunctionDef tree = (FunctionDef) ctx.syntaxNode();
          ctx.addIssue(tree.name(), tree.name().firstToken().value());
        });
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(FILE, check);

    assertThat(issues).hasSize(2);
    PreciseIssue firstIssue = issues.get(0);

    assertThat(firstIssue.cost()).isNull();
    assertThat(firstIssue.secondaryLocations()).isEmpty();

    IssueLocation primaryLocation = firstIssue.primaryLocation();
    assertThat(primaryLocation.message()).isEqualTo("hello");

    assertThat(primaryLocation.startLine()).isEqualTo(1);
    assertThat(primaryLocation.endLine()).isEqualTo(1);
    assertThat(primaryLocation.startLineOffset()).isEqualTo(4);
    assertThat(primaryLocation.endLineOffset()).isEqualTo(9);
  }

  @Test
  void test_cost() {
    TestPythonCheck check = new TestPythonCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
          FunctionDef pyFunctionDefTree = (FunctionDef) ctx.syntaxNode();
          Name name = pyFunctionDefTree.name();
          ctx.addIssue(name.firstToken(), MESSAGE).withCost(42);
        });
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(FILE, check);
    PreciseIssue firstIssue = issues.get(0);
    assertThat(firstIssue.cost()).isEqualTo(42);
  }

  @Test
  void test_tokens() {
    TestPythonCheck check = new TestPythonCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
          Token pyToken = (Token) ctx.syntaxNode();
          if (pyToken.value().equals("def")) {
            ctx.addIssue(pyToken, MESSAGE);
          }
        });
      }
    };
    List<PreciseIssue> issues = scanFileForIssues(FILE, check);
    assertThat(issues).hasSize(2);
    assertThat(issues.get(0).primaryLocation().startLine()).isEqualTo(1);
    assertThat(issues.get(1).primaryLocation().startLine()).isEqualTo(7);
  }

  @Test
  void test_trivia() {
    TestPythonCheck check = new TestPythonCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
          Token token = (Token) ctx.syntaxNode();
          for (Trivia trivia : token.trivia()) {
            ctx.addIssue(trivia.token(), MESSAGE);
          }
        });
      }
    };
    List<PreciseIssue> issues = scanFileForIssues(FILE, check);
    assertThat(issues).hasSize(1);
    IssueLocation primaryLocation = issues.get(0).primaryLocation();
    assertThat(primaryLocation.startLine()).isEqualTo(5);
    assertThat(primaryLocation.endLine()).isEqualTo(5);
    assertThat(primaryLocation.startLineOffset()).isEqualTo(0);
    assertThat(primaryLocation.endLineOffset()).isEqualTo(10);
  }

  @Test
  void test_file_line() {
    TestPythonCheck check = new TestPythonCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.PASS_STMT, ctx -> {
          ctx.addLineIssue(MESSAGE, ctx.syntaxNode().firstToken().line());
          ctx.addFileIssue(MESSAGE);
        });
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(FILE, check);
    assertThat(issues).hasSize(2);
    assertThat(issues.get(0).primaryLocation().startLine()).isEqualTo(8);
  }

  private abstract static class TestPythonCheck extends PythonSubscriptionCheck {

  }
}
