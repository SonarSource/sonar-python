/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.plugins.python.pylint;

import org.junit.Rule;
import org.junit.Test;
import org.sonar.api.utils.log.LogTester;

import static org.assertj.core.api.Assertions.assertThat;

public class PylintRuleParserTest {

  private static final String NO_RULE_FOUND_MESSAGE = "No rule key found for Pylint";

  @Rule
  public LogTester logTester = new LogTester();

  @Test
  public void hasExpectedRules() {
    PylintRuleParser pylintRuleParser = new PylintRuleParser(PylintRuleRepository.RULES_FILE);
    assertThat(pylintRuleParser.hasRuleDefinition("C0102")).isTrue();
    assertThat(pylintRuleParser.hasRuleDefinition("C9999")).isFalse();
  }

  @Test
  public void logsWhenEmpty() {
    new PylintRuleParser("/org/sonar/plugins/python/pylint/empty.xml");
    assertThat(logTester.logs()).containsExactly(NO_RULE_FOUND_MESSAGE);
  }

  @Test
  public void logsWhenFileNotFound() {
    new PylintRuleParser("/org/sonar/plugins/python/pylint/no-file.xml");
    assertThat(logTester.logs()).containsExactly("Unable to parse the Pylint rules definition XML file", NO_RULE_FOUND_MESSAGE);
  }

  @Test
  public void logsWhenException() {
    new PylintRuleParser("/org/sonar/plugins/python/pylint/pylint-report.txt");
    assertThat(logTester.logs()).containsExactly("Unable to parse the Pylint rules definition XML file", NO_RULE_FOUND_MESSAGE);
  }

}
