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
package org.sonar.python;

import java.io.File;
import org.junit.Test;
import org.sonar.python.metrics.FileLinesVisitor;

import static org.assertj.core.api.Assertions.assertThat;

public class FileLinesVisitorTest {

  private static final File BASE_DIR = new File("src/test/resources/metrics");

  @Test
  public void test() {
    FileLinesVisitor visitor = new FileLinesVisitor(false);

    TestPythonVisitorRunner.scanFile(new File(BASE_DIR, "file_lines.py"), visitor);

    assertThat(visitor.getLinesOfCode()).hasSize(12);
    assertThat(visitor.getLinesOfCode()).containsOnly(2, 4, 7, 8, 9, 10, 11, 12, 14, 15, 17, 21);

    assertThat(visitor.getLinesOfComments()).hasSize(9);
    assertThat(visitor.getLinesOfComments()).containsOnly(1, 4, 6, 13, 14, 17, 18, 19, 20);

    assertThat(visitor.getLinesWithNoSonar()).containsOnly(11);
  }

  @Test
  public void test_ignoreHeaderComments() {
    FileLinesVisitor visitor = new FileLinesVisitor(true);

    TestPythonVisitorRunner.scanFile(new File(BASE_DIR, "file_lines_header_comments.py"), visitor);

    assertThat(visitor.getLinesOfCode()).containsOnly(2, 4);
    assertThat(visitor.getLinesOfComments()).containsOnly(4);
  }

  @Test
  public void executable_lines() throws Exception {
    FileLinesVisitor visitor = new FileLinesVisitor(false);
    TestPythonVisitorRunner.scanFile(new File(BASE_DIR, "executable_lines.py"), visitor);
    assertThat(visitor.getExecutableLines()).containsOnly(1, 2, 4, 7, 11, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 27, 28, 29);
  }

}
