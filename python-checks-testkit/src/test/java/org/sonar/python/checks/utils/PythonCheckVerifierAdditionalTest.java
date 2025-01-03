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
package org.sonar.python.checks.utils;

import java.util.Collections;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;

class PythonCheckVerifierAdditionalTest {

  private static final String BASE_DIR = "src/test/resources/";

  @Test
  void file_level_issue() {
    PythonCheckVerifier.verify(BASE_DIR + "file_issue.py", new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> ctx.addFileIssue("This file has a function."));
      }
    });
  }

  @Test
  void line_level_issue() {
    PythonCheckVerifier.verify(BASE_DIR + "line_issue.py", new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> ctx.addLineIssue("This line has a function.", ctx.syntaxNode().firstToken().line()));
      }
    });
  }

  @Test
  void no_issue() {
    PythonSubscriptionCheck check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> ctx.addIssue(ctx.syntaxNode().firstToken(), "This is a function"));
      }
    };
    PythonCheckVerifier.verifyNoIssue(BASE_DIR + "no_issue.py", check);
    PythonCheckVerifier.verifyNoIssue(Collections.singletonList(BASE_DIR + "no_issue.py"), check);

    try {
      PythonCheckVerifier.verifyNoIssue(BASE_DIR + "file_issue.py", check);
    } catch (AssertionError e) {
      return;
    }
    Assertions.fail("Should have failed");
  }

  @Test
  void issues_with_issue() {
    var issues = PythonCheckVerifier.issues(BASE_DIR + "line_issue.py", new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> ctx.addLineIssue("This line has a function.", ctx.syntaxNode().firstToken().line()));
      }
    });

    assertThat(issues).hasSize(1);
  }

  @Test
  void issues_without_issue() {
    var issues = PythonCheckVerifier.issues(BASE_DIR + "no_issue.py", new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> ctx.addLineIssue("This line has a function.", ctx.syntaxNode().firstToken().line()));
      }
    });

    assertThat(issues).isEmpty();
  }
}
