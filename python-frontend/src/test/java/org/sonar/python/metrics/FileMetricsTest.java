/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.metrics;

import java.io.File;
import org.junit.jupiter.api.Test;
import org.sonar.python.TestPythonVisitorRunner;

import static org.fest.assertions.Assertions.assertThat;

class FileMetricsTest {

  @Test
  void statements() {
    assertThat(metrics("statements.py").numberOfStatements()).isEqualTo(1);
  }

  @Test
  void functions() {
    assertThat(metrics("functions.py").numberOfFunctions()).isEqualTo(1);
  }

  @Test
  void classes() {
    assertThat(metrics("classes.py").numberOfClasses()).isEqualTo(1);
  }

  @Test
  void complexity() {
    assertThat(metrics("complexity.py").complexity()).isEqualTo(8);
  }

  @Test
  void cognitive_complexity() {
    assertThat(metrics("classes.py").cognitiveComplexity()).isEqualTo(0);
    assertThat(metrics("cognitive-complexities.py").cognitiveComplexity()).isEqualTo(91);
  }

  @Test
  void function_complexities() {
    assertThat(metrics("function-complexities.py").functionComplexities()).containsExactly(3, 1);
  }

  private static FileMetrics metrics(String fileName) {
    File baseDir = new File("src/test/resources/metrics/");
    File file = new File(baseDir, fileName);
    return new FileMetrics(TestPythonVisitorRunner.createContext(file));
  }

}
