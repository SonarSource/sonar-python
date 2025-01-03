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
package org.sonar.plugins.python.xunit;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a unit test suite. Contains testcases, maintains some statistics. Reports testcase details in sonar-conform XML
 */
public class TestSuite {

  private String key;
  private List<TestCase> testCases;

  /**
   * Creates a testsuite instance uniquely identified by the given key
   * 
   * @param key
   *          The key to construct a testsuite for
   */
  public TestSuite(String key) {
    this.key = key;
    this.testCases = new ArrayList<>();
  }

  public String getKey() {
    return key;
  }

  public void addTestCase(TestCase tc) {
    testCases.add(tc);
  }

  public List<TestCase> getTestCases() {
    return testCases;
  }

  /**
   * Returns execution details as sonar-conform XML
   */
  public String getDetails() {
    StringBuilder details = new StringBuilder();
    details.append("<tests-details>");
    for (TestCase tc : testCases) {
      details.append(tc.getDetails());
    }
    details.append("</tests-details>");
    return details.toString();
  }

}
