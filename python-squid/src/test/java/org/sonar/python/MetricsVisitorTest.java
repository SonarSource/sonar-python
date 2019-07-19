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
package org.sonar.python;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.sonar.api.internal.google.common.io.Files;
import org.sonar.python.metrics.MetricsVisitor;

import static org.assertj.core.api.Assertions.assertThat;

public class MetricsVisitorTest {

  private static final File BASE_DIR = new File("src/test/resources/metrics");

  @Test
  public void test() {
    MetricsVisitor visitor = metricsVisitor(new File(BASE_DIR, "file_lines.py"), false);

    assertThat(visitor.getLinesOfCode()).hasSize(12);
    assertThat(visitor.getLinesOfCode()).containsOnly(2, 4, 7, 8, 9, 10, 11, 12, 14, 15, 17, 21);

    assertThat(visitor.getCommentLineCount()).isEqualTo(9);

    assertThat(visitor.getLinesWithNoSonar()).containsOnly(11);
  }

  @Test
  public void test_ignoreHeaderComments() {
    // do not ignoreHeaderComments
    MetricsVisitor visitor = metricsVisitor(new File(BASE_DIR, "file_lines_header_comments.py"), false);
    assertThat(visitor.getLinesOfCode()).containsOnly(6, 8);
    assertThat(visitor.getCommentLineCount()).isEqualTo(4);

    // ignoreHeaderComments
    visitor = metricsVisitor(new File(BASE_DIR, "file_lines_header_comments.py"), true);
    assertThat(visitor.getLinesOfCode()).containsOnly(6, 8);
    assertThat(visitor.getCommentLineCount()).isEqualTo(1);
  }

  @Test
  public void executable_lines() {
    MetricsVisitor visitor = metricsVisitor(new File(BASE_DIR, "executable_lines.py"), false);

    assertThat(visitor.getExecutableLines()).containsOnly(1, 2, 4, 7, 11, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 27, 28, 29);
  }

  private static MetricsVisitor metricsVisitor(File file, boolean ignoreHeaderComments) {
    MetricsVisitor visitor = new MetricsVisitor(ignoreHeaderComments);
    String fileContent;
    try {
      fileContent = Files.toString(file, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    new org.sonar.python.frontend.PythonParser().parse(fileContent).accept(visitor);
    return visitor;
  }

}
