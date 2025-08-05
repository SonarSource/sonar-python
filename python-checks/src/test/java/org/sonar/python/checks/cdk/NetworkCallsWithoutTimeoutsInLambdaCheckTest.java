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
package org.sonar.python.checks.cdk;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class NetworkCallsWithoutTimeoutsInLambdaCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify(
      List.of("src/test/resources/checks/networkCallsWithoutTimeoutsInLambda/with_lambda_handler.py"), 
      new NetworkCallsWithoutTimeoutsInLambdaCheck()
    );
  }

  @Test
  void test_no_lambda_handler() {
    PythonCheckVerifier.verifyNoIssue(
      List.of("src/test/resources/checks/networkCallsWithoutTimeoutsInLambda/without_lambda_handler.py"), 
      new NetworkCallsWithoutTimeoutsInLambdaCheck()
    );
  }

}
