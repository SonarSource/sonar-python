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
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.fail;

class PythonCheckVerifierTest {

  private static final String BASE_DIR = "src/test/resources/";
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

  public static Stream<Arguments> data() {
    return Stream.of(
      Arguments.of(BASE_DIR + "compliant.py", true),
      Arguments.of(BASE_DIR + "compliant_notation.py", true),
      Arguments.of(BASE_DIR + "compliant_notation_with_minus.py", true),
      Arguments.of(BASE_DIR + "missing_assertion.py", false),
      Arguments.of(BASE_DIR + "missing_assertion_with_issue.py", false),
      Arguments.of(BASE_DIR + "missing_issue.py", false),
      Arguments.of(BASE_DIR + "missing_issue_multiple.py", false),
      Arguments.of(BASE_DIR + "unexpected_issue.py", false),
      Arguments.of(BASE_DIR + "unexpected_issue_multiple.py", false),
      Arguments.of(BASE_DIR + "wrong_cost.py", false)
    );
  }

  @ParameterizedTest
  @MethodSource("data")
  void basetree_test(String file, boolean expectSuccess) {
    if(expectSuccess) {
      assertNoFailureOfVerifier(file, baseTreeCheck);
    } else {
      assertFailOfVerifier(file, baseTreeCheck);
    }
  }

  @ParameterizedTest
  @MethodSource("data")
  void subscription_test(String file, boolean expectSuccess) {
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
