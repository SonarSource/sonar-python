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
package org.sonar.plugins.python.xunit;

import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class TestSuiteTest {
  TestSuite suite;
  TestSuite equalSuite;
  TestSuite otherSuite;

  @Before
  public void setUp() {
    suite = new TestSuite("key");
    equalSuite = new TestSuite("key");
    otherSuite = new TestSuite("otherkey");
  }

  @Test
  public void suiteDoesntEqualsNull() {
    assertThat(suite).isNotEqualTo(null);
  }

  @Test
  public void suiteDoesntEqualsMiscObject() {
    assertThat(suite).isNotEqualTo("string");
  }

  @Test
  public void suiteEqualityIsReflexive() {
    assertThat(suite).isEqualTo(suite);
    assertThat(otherSuite).isEqualTo(otherSuite);
    assertThat(equalSuite).isEqualTo(equalSuite);
  }

  @Test
  public void suiteEqualityWorksAsExpected() {
    assertThat(suite).isEqualTo(equalSuite);
    assertThat(suite).isNotEqualTo(otherSuite);
  }

  @Test
  public void suiteHashWorksAsExpected() {
    assertThat(suite.hashCode()).isEqualTo(equalSuite.hashCode());
    assertThat(suite.hashCode()).isNotEqualTo(otherSuite.hashCode());
  }

  @Test
  public void newBornSuiteShouldHaveVirginStatistics() {
    assertThat(suite.getTests()).isEqualTo(0);
    assertThat(suite.getErrors()).isEqualTo(0);
    assertThat(suite.getFailures()).isEqualTo(0);
    assertThat(suite.getSkipped()).isEqualTo(0);
    assertThat(suite.getTime()).isEqualTo(0);
    assertThat(suite.getDetails()).isEqualTo("<tests-details></tests-details>");
  }

  @Test
  public void addingTestCaseShouldIncrementStatistics() {
    int testBefore = suite.getTests();
    int timeBefore = suite.getTime();

    final int EXEC_TIME = 10;
    suite.addTestCase(new TestCase("name", EXEC_TIME, "status", "stack", "msg"));

    assertThat(suite.getTests()).isEqualTo(testBefore + 1);
    assertThat(suite.getTime()).isEqualTo(timeBefore + EXEC_TIME);
  }

  @Test
  public void addingAnErroneousTestCaseShouldIncrementErrorStatistic() {
    int errorsBefore = suite.getErrors();
    TestCase error = mock(TestCase.class);
    when(error.isError()).thenReturn(true);

    suite.addTestCase(error);

    assertThat(suite.getErrors()).isEqualTo(errorsBefore + 1);
  }

  @Test
  public void addingAFailedestCaseShouldIncrementFailedStatistic() {
    int failedBefore = suite.getFailures();
    TestCase failedTC = mock(TestCase.class);
    when(failedTC.isFailure()).thenReturn(true);

    suite.addTestCase(failedTC);

    assertThat(suite.getFailures()).isEqualTo(failedBefore + 1);
  }

  @Test
  public void addingASkippedTestCaseShouldIncrementSkippedStatistic() {
    int skippedBefore = suite.getSkipped();
    TestCase skippedTC = mock(TestCase.class);
    when(skippedTC.isSkipped()).thenReturn(true);

    suite.addTestCase(skippedTC);

    assertThat(suite.getSkipped()).isEqualTo(skippedBefore + 1);
  }

  @Test
  public void addingAnotherTestSuiteShouldMaintainStatistics() {
    TestCase tc = mock(TestCase.class);
    when(tc.isSkipped()).thenReturn(true);
    TestSuite ts1 = new TestSuite("1");
    TestSuite ts2 = new TestSuite("2");
    ts1.addTestCase(tc);
    ts2.addTestCase(tc);

    TestSuite summedUp = ts1.addMeasures(ts2);

    assertThat(summedUp.getSkipped()).isEqualTo(2);
  }
}
