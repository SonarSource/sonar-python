/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.regex;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.ast.CharacterTree;
import org.sonarsource.analyzer.commons.regex.ast.RegexSyntaxElement;
import org.sonarsource.analyzer.commons.regex.ast.RegexTree;
import org.sonarsource.analyzer.commons.regex.ast.SequenceTree;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.regex.RegexParserTestUtils.assertKind;
import static org.sonar.python.regex.RegexParserTestUtils.assertSuccessfulParse;

class PythonRegexIssueLocationTest {

  @Test
  void test_regex_locations_to_issue_locations() {
    // force a separation
    RegexTree regex = assertSuccessfulParse("r'ab'");
    assertKind(RegexTree.Kind.SEQUENCE, regex);

    List<RegexTree> items = ((SequenceTree) regex).getItems();
    assertThat(items).hasSize(2)
      .allMatch(tree -> tree.is(RegexTree.Kind.CHARACTER));

    assertRange(11, 13, correspondingPythonIssueLocation(regex));
    CharacterTree char1 = (CharacterTree) items.get(0);
    assertRange(11, 12, correspondingPythonIssueLocation(char1));
    CharacterTree char2 = (CharacterTree) items.get(1);
    assertRange(12, 13, correspondingPythonIssueLocation(char2));
  }

  @Test
  void test_location_of_multiple_regex_syntax_element() {
    // force a separation
    RegexTree regex = assertSuccessfulParse("r'ABC'");
    assertKind(RegexTree.Kind.SEQUENCE, regex);

    List<RegexTree> items = ((SequenceTree) regex).getItems();
    assertThat(items).hasSize(3);

    RegexTree A = items.get(0);
    RegexTree B = items.get(1);
    RegexTree C = items.get(2);

    assertRange(11, 14, correspondingPythonIssueLocation(Arrays.asList(A, B, C)));
    assertRange(11, 13, correspondingPythonIssueLocation(Arrays.asList(A, B)));
    assertRange(11, 12, correspondingPythonIssueLocation(Arrays.asList(A, C)));
  }

  @Test
  void test_location_of_regex_issue_location() {
    // force a separation
    RegexTree regex = assertSuccessfulParse("r'A'");
    RegexIssueLocation location = new RegexIssueLocation(regex, "message");
    assertRange(11, 12, correspondingPythonIssueLocation(location));
  }

  private void assertRange(int startLineOffset, int endLineOffset, IssueLocation location) {
    assertThat(location.startLineOffset()).withFailMessage(String.format("Expected start character to be '%d' but got '%d'", startLineOffset, location.startLineOffset())).isEqualTo(startLineOffset);
    assertThat(location.endLineOffset()).withFailMessage(String.format("Expected end character to be '%d' but got '%d'", endLineOffset, location.endLineOffset())).isEqualTo(endLineOffset);
  }

  private static IssueLocation correspondingPythonIssueLocation(RegexIssueLocation location) {
    return PythonRegexIssueLocation.preciseLocation(location);
  }

  private static IssueLocation correspondingPythonIssueLocation(RegexTree tree) {
    return PythonRegexIssueLocation.preciseLocation(tree, "message");
  }

  private static IssueLocation correspondingPythonIssueLocation(List<RegexSyntaxElement> trees) {
    return PythonRegexIssueLocation.preciseLocation(trees, "message");
  }

}
