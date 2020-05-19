/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.checks;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class DictionaryDuplicateKeyCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/dictionaryDuplicateKey.py", new DictionaryDuplicateKeyCheck());
  }

  /** an expression with a single big dictionary literal. The literal is multiline. */
  private static String dictionaryLiteral(int numSyntheticEntries, String... extraLinesInside) {
    StringBuilder contentBuilder = new StringBuilder();
    contentBuilder.append("{\n");
    for (int i = 0; i < numSyntheticEntries; i++) {
      // note that dangling comma is ok
      contentBuilder.append("  'syntheticEntry_" + i + "': " + i + ",\n");
    }
    for (String a: extraLinesInside) {
      contentBuilder.append("  " + a + "\n");
    }
    contentBuilder.append("}\n");
    return contentBuilder.toString();
  }

  @Test(timeout = 2000L)
  public void tinySyntheticDictionaryTest() {
    PythonCheckVerifier.verifyTemp(
      dictionaryLiteral(10, "'a': 1, # Noncompliant", "'a': 2"),
      new DictionaryDuplicateKeyCheck());
  }

  @Test(timeout = 2000L)
  public void smallSyntheticDictionaryTest() {
    PythonCheckVerifier.verifyTemp(
      dictionaryLiteral(97, "'a': 1, # Noncompliant", "'a': 2"),
      new DictionaryDuplicateKeyCheck());
  }


  // @Test(timeout = 5000L)
  public void hugeSyntheticDictionaryTest() {
    PythonCheckVerifier.verifyTemp(
      dictionaryLiteral(10000, "'a': 1, # Noncompliant", "'a': 2"),
      new DictionaryDuplicateKeyCheck());
  }
}
