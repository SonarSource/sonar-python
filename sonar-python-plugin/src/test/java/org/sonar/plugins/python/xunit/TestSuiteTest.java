/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python.xunit;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class TestSuiteTest {

  @Test
  void test() {
    TestSuite suite = new TestSuite("key");
    assertThat(suite.getKey()).isEqualTo("key");
    assertThat(suite.getTestCases()).isEmpty();
    assertThat(suite.getDetails()).isEqualTo("<tests-details></tests-details>");

    TestCase testCase = new TestCase("name", 1, "status", "stack", "msg", "file", "testClassname");
    suite.addTestCase(testCase);
    assertThat(suite.getTestCases()).containsExactly(testCase);
    assertThat(suite.getDetails()).isEqualTo("<tests-details><testcase status=\"status\" time=\"1\" name=\"name\"/></tests-details>");
  }

}
