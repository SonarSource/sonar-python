/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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


import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCheckTree;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

@RunWith(Parameterized.class)
public class PythonCheckVerifierTest {

  private static final String BASE_DIR = "src/test/resources/";
  private final String file;
  private final String expectedMessage;
  private static final FuncdefVisitor baseTreeCheck = new FuncdefVisitor();
  private static final FunctiondefSubscription subscriptionCheck = new FunctiondefSubscription();

  private static class FuncdefVisitor extends PythonCheckTree {
    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      addIssue(pyFunctionDefTree.name(), "the message.").secondary(pyFunctionDefTree.colon(), "second").withCost(42);
      super.visitFunctionDef(pyFunctionDefTree);
    }
  }

  private static class FunctiondefSubscription extends PythonSubscriptionCheck {
    @Override
    public void initialize(Context context) {
      context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
        FunctionDef pyFunctionDefTree = (FunctionDef) ctx.syntaxNode();
        ctx.addIssue(pyFunctionDefTree.name(), "the message.").secondary(pyFunctionDefTree.colon(), "second").withCost(42);
      });
    }
  }

  @Parameters(name = "{0}")
  public static Object[][] data() {
    return new Object[][]{
      {"compliant", ""},
      {"compliant_notation", ""},
      {"compliant_notation_with_minus", ""},
      {"invalid_param", "Invalid param at line 1: someInvalidParam"},
      {"invalid_param_separator", "Invalid param at line 1: someInvalidParam!!!!!42"},
      {"missing_assertion", "Invalid test file: a precise location is provided at line 2 but no issue is asserted at line 1"},
      {"missing_assertion_with_issue", "Invalid test file: a precise location is provided at line 4 but no issue is asserted at line 3"},
      {"missing_issue", "Missing issue at line 1"},
      {"missing_issue_multiple", "Missing issue at line 1"},
      {"unexpected_issue", "Unexpected issue at line 1: \"the message.\""},
      {"unexpected_issue_multiple", "Unexpected issue at line 1: \"the message.\""},
      {"wrong_cost", "[Bad effortToFix at line 1] expected:<[23]> but was:<[42]>"},
      {"wrong_precise_comment", "Line 2: comments asserting a precise location should start at column 1"},
    };
  }

  public PythonCheckVerifierTest(String file, String expectedMessage) {
    this.file = BASE_DIR+file+".py";
    this.expectedMessage = expectedMessage;
  }

  @Test
  public void basetree_test() {
    if(expectedMessage.isEmpty()) {
      assertNoFailureOfVerifier(file, baseTreeCheck);
    } else {
      assertFailOfVerifier(file, expectedMessage, baseTreeCheck);
    }
  }

  @Test
  public void subscription_test() {
    if(expectedMessage.isEmpty()) {
      assertNoFailureOfVerifier(file, subscriptionCheck);
    } else {
      assertFailOfVerifier(file, expectedMessage, subscriptionCheck);
    }
  }

  private void assertNoFailureOfVerifier(String filePath, PythonCheck check) {
    try {
      PythonCheckVerifier.verify(filePath, check);
    } catch (AssertionError e) {
      fail("should not fail", e);
    }
  }

  private static void assertFailOfVerifier(String filepath, String expectedFailureMessage, PythonCheck check) {
    try {
      PythonCheckVerifier.verify(filepath, check);
      fail("should have failed");
    } catch (AssertionError | IllegalStateException e) {
      assertThat(e.getMessage()).isEqualTo(expectedFailureMessage);
    }
  }
}
