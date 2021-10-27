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
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;

import static org.assertj.core.api.Assertions.assertThat;

public class AbstractRegexCheckTest {

  private static final File FILE = new File("src/test/resources/checks/regex/abstractRegexCheck.py");

  private static PythonVisitorContext fileContext(File file) {
    return TestPythonVisitorRunner.createContext(file);
  }
  
  private static List<PythonCheck.PreciseIssue> scanFileForIssues(PythonVisitorContext fileContext, List<PythonSubscriptionCheck> checks) {
    checks.forEach(c -> c.scanFile(fileContext));
    SubscriptionVisitor.analyze(checks, fileContext);
    return fileContext.getIssues();
  }

  @Test
  public void test_regex_is_visited() {
    Check check = new Check();
    List<PythonCheck.PreciseIssue> issues = scanFileForIssues(fileContext(FILE), Collections.singletonList(check));
    assertThat(check.receivedRegexParseResults).hasSize(1);
    assertThat(issues).hasSize(1);
  }

  @Test
  public void test_regex_parse_result_is_retrieved_from_cache_in_context() {
    PythonVisitorContext fileContext = fileContext(FILE);

    Check checkOne = new Check();
    Check checkTwo = new Check();
    scanFileForIssues(fileContext, Arrays.asList(checkOne, checkTwo));

    assertThat(checkOne.receivedRegexParseResults.get(0)).isSameAs(checkTwo.receivedRegexParseResults.get(0));
  }

  private static class Check extends AbstractRegexCheck {
    private final List<RegexParseResult> receivedRegexParseResults = new ArrayList<>();

    public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
      receivedRegexParseResults.add(regexParseResult);
      regexContext.addIssue(regexFunctionCall, "MESSAGE");
    }
  }

}