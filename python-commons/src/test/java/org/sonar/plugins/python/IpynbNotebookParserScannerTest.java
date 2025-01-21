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
package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.checks.TrailingWhitespaceCheck;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.TestUtils.createInputFile;

class IpynbNotebookParserScannerTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python");

  @Test
  void trailing_whitespace() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_trailing_whitespace.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    var result = IpynbNotebookParser.parseNotebook(inputFile).get();
    var check = new TrailingWhitespaceCheck();
    var context = new PythonVisitorContext(
      TestPythonVisitorRunner.parseNotebookFile(result.locationMap(), result.contents()),
      SonarQubePythonFile.create(result),
      null,
      "");
    check.scanFile(context);

    var issues = context.getIssues();
    assertThat(issues).hasSize(3);

    assertThat(issues.get(0).primaryLocation().startLine()).isEqualTo(14);
    assertThat(issues.get(0).primaryLocation().endLine()).isEqualTo(14);
    assertThat(issues.get(0).primaryLocation().startLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
    assertThat(issues.get(0).primaryLocation().endLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);

    assertThat(issues.get(1).primaryLocation().startLine()).isEqualTo(17);
    assertThat(issues.get(1).primaryLocation().endLine()).isEqualTo(17);
    assertThat(issues.get(1).primaryLocation().startLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
    assertThat(issues.get(1).primaryLocation().endLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);

    assertThat(issues.get(2).primaryLocation().startLine()).isEqualTo(20);
    assertThat(issues.get(2).primaryLocation().endLine()).isEqualTo(20);
    assertThat(issues.get(2).primaryLocation().startLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
    assertThat(issues.get(2).primaryLocation().endLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
  }

  @Test
  void trailing_whitespace_compressed() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_trailing_whitespace_compressed.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    var result = IpynbNotebookParser.parseNotebook(inputFile).get();
    var check = new TrailingWhitespaceCheck();
    var context = new PythonVisitorContext(
      TestPythonVisitorRunner.parseNotebookFile(result.locationMap(), result.contents()),
      SonarQubePythonFile.create(result),
      null,
      "");
    check.scanFile(context);

    var issues = context.getIssues();
    assertThat(issues).hasSize(3);

    assertThat(issues.get(0).primaryLocation().startLine()).isEqualTo(1);
    assertThat(issues.get(0).primaryLocation().endLine()).isEqualTo(1);
    assertThat(issues.get(0).primaryLocation().startLineOffset()).isEqualTo(142);
    assertThat(issues.get(0).primaryLocation().endLineOffset()).isEqualTo(157); // Should be 154

    assertThat(issues.get(1).primaryLocation().startLine()).isEqualTo(1);
    assertThat(issues.get(1).primaryLocation().endLine()).isEqualTo(1);
    assertThat(issues.get(1).primaryLocation().startLineOffset()).isEqualTo(199);
    assertThat(issues.get(1).primaryLocation().endLineOffset()).isEqualTo(207); // Should be 204

    assertThat(issues.get(2).primaryLocation().startLine()).isEqualTo(1);
    assertThat(issues.get(2).primaryLocation().endLine()).isEqualTo(1);
    assertThat(issues.get(2).primaryLocation().startLineOffset()).isEqualTo(249);
    assertThat(issues.get(2).primaryLocation().endLineOffset()).isEqualTo(249); // Should be 255
  }

}
