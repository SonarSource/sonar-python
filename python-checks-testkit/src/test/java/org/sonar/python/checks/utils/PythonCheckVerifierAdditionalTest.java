/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.checks.utils;

import java.util.Collections;
import org.assertj.core.api.Assertions;
import org.junit.Test;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Tree;

public class PythonCheckVerifierAdditionalTest {

  private static final String BASE_DIR = "src/test/resources/";

  @Test
  public void file_level_issue() {
    PythonCheckVerifier.verify(BASE_DIR + "file_issue.py", new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> ctx.addFileIssue("This file has a function."));
      }
    });
  }

  @Test
  public void line_level_issue() {
    PythonCheckVerifier.verify(BASE_DIR + "line_issue.py", new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> ctx.addLineIssue("This line has a function.", ctx.syntaxNode().firstToken().line()));
      }
    });
  }

  @Test
  public void no_issue() {
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
}
