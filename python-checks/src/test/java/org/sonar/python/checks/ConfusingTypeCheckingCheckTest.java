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
package org.sonar.python.checks;

import java.io.File;
import java.util.Collections;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class ConfusingTypeCheckingCheckTest {

  private final ConfusingTypeCheckingCheck check = new ConfusingTypeCheckingCheck();

  @Test
  void non_callable_called() {
    PythonCheckVerifier.verify("src/test/resources/checks/confusingTypeChecking/nonCallableCalled.py", check);
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/nonCallableCalled.py");
  }

  @Test
  void incompatible_operands() {
    PythonCheckVerifier.verify("src/test/resources/checks/confusingTypeChecking/incompatibleOperands.py", check);
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/incompatibleOperands/comparison.py");
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/incompatibleOperands/arithmetic.py");
  }

  @Test
  void item_operations() {
    PythonCheckVerifier.verify("src/test/resources/checks/confusingTypeChecking/itemOperations.py", check);
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/itemOperationsTypeCheck/itemOperations_delitem.py");
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/itemOperationsTypeCheck/itemOperations_getitem.py");
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/itemOperationsTypeCheck/itemOperations_setitem.py");
  }

  @Test
  void iteration_on_non_iterable() {
    PythonCheckVerifier.verify("src/test/resources/checks/confusingTypeChecking/iterationOnNonIterable.py", check);
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/iterationOnNonIterable.py");
  }

  @Test
  void incorrect_exception_type() {
    PythonCheckVerifier.verify("src/test/resources/checks/confusingTypeChecking/incorrectExceptionType.py", check);
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/incorrectExceptionType/incorrectExceptionType.py");
  }

  @Test
  void silly_equality() {
    PythonCheckVerifier.verify("src/test/resources/checks/confusingTypeChecking/sillyEquality.py", check);
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/sillyEquality.py");
  }

  @Test
  void silly_identity() {
    PythonCheckVerifier.verify("src/test/resources/checks/confusingTypeChecking/sillyIdentity.py", check);
    assertNoIssuesInCorrespondingBugRule("src/test/resources/checks/sillyIdentity.py");
  }

  private void assertNoIssuesInCorrespondingBugRule(String path) {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(new File(path));
    SubscriptionVisitor.analyze(Collections.singletonList(check), context);
    assertThat(context.getIssues()).isEmpty();
  }
}
