/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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

import com.sonar.sslr.api.Grammar;
import java.io.File;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Set;
import org.junit.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.python.api.PythonMetric;
import org.sonar.python.metrics.FileLinesVisitor;
import org.sonar.squidbridge.SquidAstVisitor;
import org.sonar.squidbridge.api.SourceFile;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

public class FileLinesVisitorTest {

  private static final File BASE_DIR = new File("src/test/resources/metrics");

  private DefaultFileSystem fileSystem = new DefaultFileSystem(Paths.get(""));

  private FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);

  private FileLinesContext fileLinesContext = mock(FileLinesContext.class);

  @Test
  public void test() {
    DefaultInputFile inputFile = initFile("file_lines.py");

    HashMap<InputFile, Set<Integer>> linesOfCode = new HashMap<>();
    SquidAstVisitor<Grammar> visitor = new FileLinesVisitor(fileLinesContextFactory, fileSystem, linesOfCode, false);

    SourceFile sourceFile = PythonAstScanner.scanSingleFile(inputFile.getFile().getPath(), visitor);

    assertThat(sourceFile.getInt(PythonMetric.LINES_OF_CODE)).isEqualTo(12);
    assertThat(linesOfCode).hasSize(1);
    assertThat(linesOfCode.get(inputFile)).as("Lines of codes").containsOnly(2, 4, 7, 8, 9, 10, 11, 12, 14, 15, 17, 21);
    verifyInvocation(fileLinesContext, CoreMetrics.NCLOC_DATA_KEY, 2, 4, 7, 8, 9, 10, 11, 12, 14, 15, 17, 21);

    assertThat(sourceFile.getInt(PythonMetric.COMMENT_LINES)).isEqualTo(9);
    verifyInvocation(fileLinesContext, CoreMetrics.COMMENT_LINES_DATA_KEY, 1, 4, 6, 13, 14, 17, 18, 19, 20);

    verify(fileLinesContext).save();
    verifyNoMoreInteractions(fileLinesContext);
  }

  @Test
  public void test_ignoreHeaderComments() {
    DefaultInputFile inputFile = initFile("file_lines_header_comments.py");

    HashMap<InputFile, Set<Integer>> linesOfCode = new HashMap<>();
    SquidAstVisitor<Grammar> visitor = new FileLinesVisitor(fileLinesContextFactory, fileSystem, linesOfCode, true);

    PythonAstScanner.scanSingleFile(inputFile.getFile().getPath(), visitor);

    assertThat(linesOfCode).hasSize(1);
    assertThat(linesOfCode.get(inputFile)).as("Lines of codes").containsOnly(2, 4);

    verifyInvocation(fileLinesContext, CoreMetrics.NCLOC_DATA_KEY, 2, 4);
    verifyInvocation(fileLinesContext, CoreMetrics.COMMENT_LINES_DATA_KEY, 4);
    verify(fileLinesContext).save();
    verifyNoMoreInteractions(fileLinesContext);
  }

  private DefaultInputFile initFile(String fileName) {
    File file = new File(BASE_DIR, fileName);
    DefaultInputFile inputFile = new DefaultInputFile("", file.getPath());
    fileSystem.add(inputFile);
    when(fileLinesContextFactory.createFor(inputFile)).thenReturn(fileLinesContext);
    return inputFile;
  }

  /**
   * Checks that method fileLinesContext.setIntValue has been invoked for the specified
   * metrics and for every specified line.
   */
  private void verifyInvocation(FileLinesContext fileLinesContext, String metric, int... lines) {
    for (int i = 0; i < lines.length; i++) {
      verify(fileLinesContext).setIntValue(metric, lines[i], 1);
    }
  }

}
