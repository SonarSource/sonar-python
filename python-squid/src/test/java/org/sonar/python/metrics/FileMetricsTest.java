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
package org.sonar.python.metrics;

import com.jetbrains.python.psi.PyFile;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.sonar.api.internal.google.common.io.Files;
import org.sonar.python.TestPythonVisitorRunner;

import static org.fest.assertions.Assertions.assertThat;

public class FileMetricsTest {

  @Test
  public void statements() {
    assertThat(metrics("statements.py").numberOfStatements()).isEqualTo(1);
  }

  @Test
  public void functions() {
    assertThat(metrics("functions.py").numberOfFunctions()).isEqualTo(1);
  }

  @Test
  public void classes() {
    assertThat(metrics("classes.py").numberOfClasses()).isEqualTo(1);
  }

  @Test
  public void complexity() {
    assertThat(metrics("complexity.py").complexity()).isEqualTo(7);
  }

  @Test
  public void cognitive_complexity() {
    assertThat(metrics("classes.py").cognitiveComplexity()).isEqualTo(0);
    assertThat(metrics("cognitive-complexities.py").cognitiveComplexity()).isEqualTo(91);
  }

  @Test
  public void function_complexities() {
    assertThat(metrics("function-complexities.py").functionComplexities()).containsExactly(3, 1);
  }

  private static FileMetrics metrics(String fileName) {
    File baseDir = new File("src/test/resources/metrics/");
    File file = new File(baseDir, fileName);
    String fileContent;
    try {
      fileContent = Files.toString(file, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    PyFile pyFile = new org.sonar.python.frontend.PythonParser().parse(fileContent);
    return new FileMetrics(TestPythonVisitorRunner.createContext(file), true, pyFile);
  }

}
