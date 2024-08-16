/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.python.metrics.FileLinesVisitor;

import static org.assertj.core.api.Assertions.assertThat;

class FileLinesVisitorTest {

  private static final File BASE_DIR = new File("src/test/resources/metrics");

  @Test
  void test() {
    FileLinesVisitor visitor = new FileLinesVisitor();

    TestPythonVisitorRunner.scanFile(new File(BASE_DIR, "file_lines.py"), visitor);

    assertThat(visitor.getLinesOfCode()).hasSize(24);
    assertThat(visitor.getLinesOfCode()).containsOnly(6, 8, 11, 12, 13, 14, 15, 16, 18, 19, 21, 25, 28, 32, 34, 36, 37, 38, 39, 40, 41, 43, 44, 45);

    assertThat(visitor.getCommentLineCount()).isEqualTo(16);

    assertThat(visitor.getLinesWithNoSonar()).containsOnly(15, 29, 30, 31, 34, 37, 38, 39, 40, 41, 45);
  }

  @Test
  void test_ignoreHeaderComments() {
    FileLinesVisitor visitor = new FileLinesVisitor();

    TestPythonVisitorRunner.scanFile(new File(BASE_DIR, "file_lines_header_comments.py"), visitor);

    assertThat(visitor.getLinesOfCode()).containsOnly(2, 4);
    assertThat(visitor.getCommentLineCount()).isEqualTo(2);
  }

  @Test
  void executable_lines() {
    FileLinesVisitor visitor = new FileLinesVisitor();
    TestPythonVisitorRunner.scanFile(new File(BASE_DIR, "executable_lines.py"), visitor);
    assertThat(visitor.getExecutableLines()).containsOnly(1, 2, 4, 7, 11, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 27, 28, 29);
  }

  @Test
  void empty_file() {
    FileLinesVisitor visitor = new FileLinesVisitor();
    TestPythonVisitorRunner.scanFile(new File(BASE_DIR, "empty.py"), visitor);
    assertThat(visitor.getExecutableLines()).isEmpty();
  }

  @Test
  void notebook_locs_single_line_file() {
    FileLinesVisitor visitor = new FileLinesVisitor(true);
    String content = """
      a = 2
      def foo():
        return 3
      """;
    var locations = Map.of(1, new IPythonLocation(1, 383, Map.of(-1, 0)),
        2, new IPythonLocation(1, 390, Map.of(-1, 0)),
        3, new IPythonLocation(1, 402, Map.of(-1, 0)), 
        4, new IPythonLocation(1, 402, Map.of(-1, 0)));
    TestPythonVisitorRunner.scanNotebookFile(new File(BASE_DIR, "notebook_locs_single_line.ipynb"), locations, content, visitor);
    assertThat(visitor.getExecutableLines()).isEmpty();
    assertThat(visitor.getLinesOfCode()).hasSize(3);
  }

  @Test
  void notebook_locs() {
    FileLinesVisitor visitor = new FileLinesVisitor(true);
    TestPythonVisitorRunner.scanFile(new File(BASE_DIR, "notebook_loc.ipynb"), visitor);
    assertThat(visitor.getExecutableLines()).isEmpty();
    assertThat(visitor.getLinesOfCode()).hasSize(17);
  }
}
