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

class FastAPIDependencyAnnotatedCheckTest {

  @Test
  void raw_candidates() {
    PythonCheckVerifier.verify(
      "src/test/resources/checks/fastAPIDependencyAnnotated.py",
      new FastAPIDependencyAnnotatedCheck(FastAPIDependencyAnnotatedCheck.IssueReportingMode.DISABLE_FILE_LOCAL_SUPPRESSION));
  }

  @Test
  void suppresses_issues_in_old_style_dominant_files() {
    PythonCheckVerifier.verifyNoIssue(
      "src/test/resources/checks/fastAPIDependencyAnnotatedFileLocalHeuristicOldStyleDominant.py",
      new FastAPIDependencyAnnotatedCheck());
  }

  @Test
  void reports_issues_in_annotated_dominant_files() {
    PythonCheckVerifier.verify(
      "src/test/resources/checks/fastAPIDependencyAnnotatedFileLocalHeuristicAnnotatedDominant.py",
      new FastAPIDependencyAnnotatedCheck());
  }

  @Test
  void reports_issues_when_old_style_sample_is_too_small_for_suppression() {
    PythonCheckVerifier.verify(
      "src/test/resources/checks/fastAPIDependencyAnnotatedFileLocalHeuristicTooSmallSample.py",
      new FastAPIDependencyAnnotatedCheck());
  }

  @Test
  void reports_issues_when_mixed_sample_is_too_small_for_suppression() {
    PythonCheckVerifier.verify(
      "src/test/resources/checks/fastAPIDependencyAnnotatedFileLocalHeuristicMixedTooSmallSample.py",
      new FastAPIDependencyAnnotatedCheck());
  }
}
