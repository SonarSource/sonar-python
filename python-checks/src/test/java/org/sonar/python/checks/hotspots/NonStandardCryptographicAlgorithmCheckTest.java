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
package org.sonar.python.checks.hotspots;

import java.util.Arrays;
import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class NonStandardCryptographicAlgorithmCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/hotspots/nonStandardCryptographicAlgorithm/nonStandardCryptographicAlgorithm.py", new NonStandardCryptographicAlgorithmCheck());
  }

  @Test
  public void test_avoid_fp_django_namespace() {
    PythonCheckVerifier.verifyNoIssue(
      Arrays.asList(
        "src/test/resources/checks/hotspots/nonStandardCryptographicAlgorithm/__init__.py",
        "src/test/resources/checks/hotspots/nonStandardCryptographicAlgorithm/django/contrib/auth/hashers.py"),
      new NonStandardCryptographicAlgorithmCheck());
  }

}
