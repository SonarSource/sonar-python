/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class CognitiveComplexityFunctionCheckTest {

  private final CognitiveComplexityFunctionCheck check = new CognitiveComplexityFunctionCheck();

  @Test
  void test() {
    check.setThreshold(0);
    PythonCheckVerifier.verify("src/test/resources/checks/cognitiveComplexityFunction.py", check);
  }

  @Test
  void default_threshold() {
    PythonCheckVerifier.verify("src/test/resources/checks/cognitiveComplexityFunctionDefault.py", check);
  }

  @Test
  void no_issue_on_django_generated_migration() {
    check.setThreshold(0);
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/django/migrations/0001_generated_migration.py", check);
  }

  @Test
  void no_issue_on_protobuf_generated_code() {
    check.setThreshold(0);
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/generated/user_pb2.py", check);
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/generated/user_pb2_grpc.py", check);
  }

  @Test
  void issue_on_protobuf_suffix_near_match() {
    check.setThreshold(0);
    PythonCheckVerifier.verify("src/test/resources/checks/generated/cognitiveComplexity_pb2_test.py", check);
  }
}
