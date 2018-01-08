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
package org.sonar.python.metrics;

import java.io.File;
import org.junit.Test;
import org.sonar.python.TestPythonVisitorRunner;

import static org.fest.assertions.Assertions.assertThat;

public class FileMetricsTest {

  @Test
  public void statements() throws Exception {
    assertThat(metrics("statements.py").numberOfStatements()).isEqualTo(1);
  }

  @Test
  public void functions() throws Exception {
    assertThat(metrics("functions.py").numberOfFunctions()).isEqualTo(1);
  }

  @Test
  public void classes() throws Exception {
    assertThat(metrics("classes.py").numberOfClasses()).isEqualTo(1);
  }

  @Test
  public void complexity() throws Exception {
    assertThat(metrics("complexity.py").complexity()).isEqualTo(7);
  }

  @Test
  public void function_complexities() throws Exception {
    assertThat(metrics("function-complexities.py").functionComplexities()).containsExactly(3, 1);
  }

  private FileMetrics metrics(String fileName) {
    File baseDir = new File("src/test/resources/metrics/");
    File file = new File(baseDir, fileName);
    return new FileMetrics(TestPythonVisitorRunner.createContext(file), true);
  }

}
