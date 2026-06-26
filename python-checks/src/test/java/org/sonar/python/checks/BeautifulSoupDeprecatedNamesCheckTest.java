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

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class BeautifulSoupDeprecatedNamesCheckTest {

  private static final BeautifulSoupDeprecatedNamesCheck check = new BeautifulSoupDeprecatedNamesCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/beautifulSoupDeprecatedNames.py", check);
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("quickFixTestCases")
  void quickFixMethod(String testName, String before, String after, String expectedMessage) {
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }


  @Test
  void quickFixTextKeyword() {
    String before = """
      from bs4 import BeautifulSoup
      soup = BeautifulSoup("<html/>", 'html.parser')
      soup.find_all('a', text='Click here')""";
    String after = """
      from bs4 import BeautifulSoup
      soup = BeautifulSoup("<html/>", 'html.parser')
      soup.find_all('a', string='Click here')""";
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace 'text' with 'string'");
  }

  static Stream<Arguments> quickFixTestCases() {
    return Stream.of(
      methodCase("findAll", "find_all"),
      methodCase("findChild", "find"),
      methodCase("findChildren", "find_all"),
      methodCase("findNext", "find_next"),
      methodCase("findAllNext", "find_all_next"),
      methodCase("findPrevious", "find_previous"),
      methodCase("findAllPrevious", "find_all_previous"),
      methodCase("findNextSibling", "find_next_sibling"),
      methodCase("findNextSiblings", "find_next_siblings"),
      methodCase("findPreviousSibling", "find_previous_sibling"),
      methodCase("findPreviousSiblings", "find_previous_siblings"),
      methodCase("findParent", "find_parent"),
      methodCase("findParents", "find_parents"),
      methodCase("replaceWith", "replace_with"),
      methodCase("getText", "get_text"),
      attrCase("nextSibling", "next_sibling"),
      attrCase("previousSibling", "previous_sibling")
    );
  }

  private static Arguments methodCase(String deprecated, String modern) {
    String before = """
      from bs4 import BeautifulSoup
      soup = BeautifulSoup("<html/>", 'html.parser')
      soup.%s('a')""".formatted(deprecated);
    String after = """
      from bs4 import BeautifulSoup
      soup = BeautifulSoup("<html/>", 'html.parser')
      soup.%s('a')""".formatted(modern);
    return Arguments.of(
      deprecated + " -> " + modern,
      before,
      after,
      "Replace '%s()' with '%s()'".formatted(deprecated, modern)
    );
  }

  private static Arguments attrCase(String deprecated, String modern) {
    String before = """
      from bs4 import BeautifulSoup
      soup = BeautifulSoup("<html/>", 'html.parser')
      _ = soup.%s""".formatted(deprecated);
    String after = """
      from bs4 import BeautifulSoup
      soup = BeautifulSoup("<html/>", 'html.parser')
      _ = soup.%s""".formatted(modern);
    return Arguments.of(
      deprecated + " -> " + modern,
      before,
      after,
      "Replace '%s' with '%s'".formatted(deprecated, modern)
    );
  }
}
