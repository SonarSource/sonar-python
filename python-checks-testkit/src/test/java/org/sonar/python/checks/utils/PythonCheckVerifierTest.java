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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.fail;

@RunWith(Parameterized.class)
public class PythonCheckVerifierTest {

  private static final String BASE_DIR = "src/test/resources/";
  private final String file;
  private final boolean expectSuccess;
  private static final FuncdefVisitor baseTreeCheck = new FuncdefVisitor();
  private static final FunctiondefSubscription subscriptionCheck = new FunctiondefSubscription();

  private static class FuncdefVisitor extends PythonVisitorCheck {
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
      {"compliant", true},
      {"compliant_notation", true},
      {"compliant_notation_with_minus", true},
      {"missing_assertion", false},
      {"missing_assertion_with_issue", false},
      {"missing_issue", false},
      {"missing_issue_multiple", false},
      {"unexpected_issue", false},
      {"unexpected_issue_multiple", false},
      {"wrong_cost", false},
    };
  }

  public PythonCheckVerifierTest(String file, boolean expectSuccess) {
    this.file = BASE_DIR+file+".py";
    this.expectSuccess = expectSuccess;
  }

  @Test
  public void basetree_test() {
    if(expectSuccess) {
      assertNoFailureOfVerifier(file, baseTreeCheck);
    } else {
      assertFailOfVerifier(file, baseTreeCheck);
    }
  }

  @Test
  public void subscription_test() {
    if(expectSuccess) {
      assertNoFailureOfVerifier(file, subscriptionCheck);
    } else {
      assertFailOfVerifier(file, subscriptionCheck);
    }
  }

  private void assertNoFailureOfVerifier(String filePath, PythonCheck check) {
    try {
      PythonCheckVerifier.verify(filePath, check);
    } catch (AssertionError e) {
      fail("should not fail", e);
    }

    try {
      PythonCheckVerifier.verify(Collections.singletonList(filePath), check);
    } catch (AssertionError e) {
      fail("should not fail", e);
    }
  }

  private static void assertFailOfVerifier(String filepath, PythonCheck check) {
    try {
      PythonCheckVerifier.verify(filepath, check);
    } catch (AssertionError | IllegalStateException e) {
      // OK, expected
      return;
    }
    fail("should have failed");
  }
}
