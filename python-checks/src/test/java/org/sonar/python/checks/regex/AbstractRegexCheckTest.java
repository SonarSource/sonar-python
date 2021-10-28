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
package org.sonar.python.checks.regex;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;

import static org.assertj.core.api.Assertions.assertThat;

public class AbstractRegexCheckTest {

  private static final File FILE = new File("src/test/resources/checks/regex/abstractRegexCheck.py");

  @Test
  public void test_regex_is_visited() {
    Check check = new Check();
    PythonCheckVerifier.verify(FILE.getAbsolutePath(), check);
    assertThat(check.receivedRegexParseResults).hasSize(10);
  }

  @Test
  public void test_regex_element_is_only_reported_once() {
    Check check = new Check() {
      @Override
      public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
        super.checkRegex(regexParseResult, regexFunctionCall);
        addIssue(regexParseResult.getResult(), "MESSAGE", null, Collections.emptyList());
      }
    };

    PythonVisitorContext fileContext = TestPythonVisitorRunner.createContext(FILE);
    SubscriptionVisitor.analyze(Collections.singletonList(check), fileContext);
    assertThat(check.reportedRegexTrees).hasSize(10);
    assertThat(fileContext.getIssues()).hasSize(10);
  }

  @Test
  public void test_regex_parse_result_is_retrieved_from_cache_in_context() {
    PythonVisitorContext fileContext = TestPythonVisitorRunner.createContext(FILE);
    Check checkOne = new Check();
    Check checkTwo = new Check();
    SubscriptionVisitor.analyze(Arrays.asList(checkOne, checkTwo), fileContext);
    assertThat(checkOne.receivedRegexParseResults).hasSameSizeAs(checkTwo.receivedRegexParseResults);
    for (int i = 0; i < checkOne.receivedRegexParseResults.size(); i++) {
      assertThat(checkOne.receivedRegexParseResults.get(i)).isSameAs(checkTwo.receivedRegexParseResults.get(i));
    }
  }

  private static class Check extends AbstractRegexCheck {
    private final List<RegexParseResult> receivedRegexParseResults = new ArrayList<>();

    @Override
    public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
      receivedRegexParseResults.add(regexParseResult);
      addIssue(regexParseResult.getResult(), "MESSAGE", null, Collections.emptyList());
    }
  }

}
