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
package org.sonar.plugins.python.xunit;

public class TestResult {

  private int errors = 0;
  private int skipped = 0;
  private int tests = 0;
  private int time = 0;
  private int failures = 0;

  public int getErrors() {
    return errors;
  }

  public int getSkipped() {
    return skipped;
  }

  public int getTests() {
    return tests;
  }

  public int getExecutedTests() {
    return tests - skipped;
  }

  public int getTime() {
    return time;
  }

  public int getFailures() {
    return failures;
  }

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
  }

}
