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
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.plugins.python.GeneratedIPythonFile;
import org.sonar.plugins.python.PythonInputFile;
import org.sonar.plugins.python.SonarQubePythonFile;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.TokenImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class IssueLocationTest {

  private static final String MESSAGE = "message";

  private PythonParser parser = PythonParser.create();

  @Test
  void file_level() {
    IssueLocation issueLocation = IssueLocation.atFileLevel(MESSAGE);
    assertThat(issueLocation.message()).isEqualTo(MESSAGE);
    assertThat(issueLocation.startLine()).isEqualTo(IssueLocation.UNDEFINED_LINE);
    assertThat(issueLocation.endLine()).isEqualTo(IssueLocation.UNDEFINED_LINE);
    assertThat(issueLocation.startLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
    assertThat(issueLocation.endLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
    assertThat(issueLocation.fileId()).isNull();
  }

  @Test
  void line_level_notebook() {
    var wrappedInputFile = TestInputFileBuilder.create("foo.ipynb", "foo").build();
    PythonInputFile pythonInputFile = new GeneratedIPythonFile(wrappedInputFile, "", Map.of(
      1, new IPythonLocation(1, 1),
      2, new IPythonLocation(2, 2),
      3, new IPythonLocation(3, 3)));
    var inputFile = SonarQubePythonFile.create(pythonInputFile);
    var issueLocation = IssueLocation.atLineLevel(MESSAGE, 2, inputFile);
    assertThat(issueLocation.message()).isEqualTo(MESSAGE);
    assertThat(issueLocation.startLine()).isEqualTo(2);
    assertThat(issueLocation.startLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
    assertThat(issueLocation.endLine()).isEqualTo(2);
    assertThat(issueLocation.endLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
  }

  @Test
  void line_level_notebook_compressed() {
    var wrappedInputFile = TestInputFileBuilder.create("foo.ipynb", "foo").build();
    PythonInputFile pythonInputFile = new GeneratedIPythonFile(wrappedInputFile, "", Map.of(
      1, new IPythonLocation(1, 1, List.of(), true),
      2, new IPythonLocation(1, 10, List.of(), true),
      3, new IPythonLocation(1, 25, List.of(), true)));
    var inputFile = SonarQubePythonFile.create(pythonInputFile);
    var issueLocation = IssueLocation.atLineLevel(MESSAGE, 2, inputFile);
    assertThat(issueLocation.message()).isEqualTo(MESSAGE);
    assertThat(issueLocation.startLine()).isEqualTo(1);
    assertThat(issueLocation.startLineOffset()).isEqualTo(10);
    assertThat(issueLocation.endLine()).isEqualTo(1);
    assertThat(issueLocation.endLineOffset()).isEqualTo(25);
  }

  @Test
  void line_level_notebook_compressed_fail() {
    var wrappedInputFile = TestInputFileBuilder.create("foo.ipynb", "foo").build();
    PythonInputFile pythonInputFile = new GeneratedIPythonFile(wrappedInputFile, "", Map.of(
      1, new IPythonLocation(1, 1, List.of(), true),
      2, new IPythonLocation(1, 10, List.of(), true)));
    var inputFile = SonarQubePythonFile.create(pythonInputFile);
    assertThatThrownBy(() -> IssueLocation.atLineLevel(MESSAGE, 2, inputFile))
      .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void precise_issue_location() {
    LocationInFile locationInFile = new LocationInFile("foo.py", 1, 1, 1, 10);
    IssueLocation issueLocation = IssueLocation.preciseLocation(locationInFile, "foo");
    assertThat(issueLocation.fileId()).isEqualTo("foo.py");
    assertThat(issueLocation.startLine()).isEqualTo(1);
    assertThat(issueLocation.startLineOffset()).isEqualTo(1);
    assertThat(issueLocation.endLine()).isEqualTo(1);
    assertThat(issueLocation.endLineOffset()).isEqualTo(10);
  }

  @Test
  void tokens() {
    AstNode root = parser.parse("\n\nfoo(42 + y) + 2");
    AstNode firstNode = root.getFirstDescendant(PythonTokenType.NUMBER);
    AstNode lastNode = root.getFirstDescendant(PythonPunctuator.RPARENTHESIS);
    IssueLocation issueLocation = IssueLocation.preciseLocation(new TokenImpl(firstNode.getToken()), new TokenImpl(lastNode.getToken()), MESSAGE);
    assertThat(issueLocation.message()).isEqualTo(MESSAGE);
    assertThat(issueLocation.startLine()).isEqualTo(3);
    assertThat(issueLocation.endLine()).isEqualTo(3);
    assertThat(issueLocation.startLineOffset()).isEqualTo(4);
    assertThat(issueLocation.endLineOffset()).isEqualTo(11);
    assertThat(issueLocation.fileId()).isNull();
  }
}
