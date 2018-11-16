/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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

import org.sonar.api.batch.fs.InputFile;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a unit test suite. Contains testcases, maintains some statistics. Reports testcase details in sonar-conform XML
 */
public class TestSuite {

  private String key;
  private InputFile inputFile = null;
  private int errors = 0;
  private int skipped = 0;
  private int tests = 0;
  private int time = 0;
  private int failures = 0;
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

  public int getErrors() {
    return errors;
  }

  public int getSkipped() {
    return skipped;
  }

  public int getTests() {
    return tests;
  }

  public int getTime() {
    return time;
  }

  public int getFailures() {
    return failures;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    TestSuite that = (TestSuite) o;
    return key.equals(that.key);
  }

  @Override
  public int hashCode() {
    return key.hashCode();
  }

  /**
   * Adds the given test case to this testsuite maintaining the internal statistics
   * 
   * @param tc
   *          the test case to add
   */
  public void addTestCase(TestCase tc) {
    if (tc.isSkipped()) {
      skipped++;
    } else if (tc.isFailure()) {
      failures++;
    } else if (tc.isError()) {
      errors++;
    }
    tests++;
    time += tc.getTime();
    testCases.add(tc);
  }

  /**
   * Adds the measures contained by the given test suite to this test suite
   * 
   * @param ts
   *          the test suite to add the measures from
   */
  public TestSuite addMeasures(TestSuite ts) {
    for (TestCase tc : ts.getTestCases()) {
      addTestCase(tc);
    }
    return this;
  }

  /**
   * Returns the testcases contained by this test suite
   */
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

  public void setInputFile(InputFile inputFile) {
    this.inputFile = inputFile;
  }

  public InputFile getInputFile() {
    return inputFile;
  }
}
