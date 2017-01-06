/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
import org.junit.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.python.metrics.FileLinesVisitor;
import org.sonar.squidbridge.SquidAstVisitor;

import java.io.File;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

public class FileLinesVisitorTest {

  private static final File BASE_DIR = new File("src/test/resources/metrics");

  @Test
  public void test() {
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    DefaultFileSystem fileSystem = new DefaultFileSystem(Paths.get(""));
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);

    File file = new File(BASE_DIR, "file_lines.py");
    DefaultInputFile inputFile = new DefaultInputFile("", file.getPath());

    fileSystem.add(inputFile);
    when(fileLinesContextFactory.createFor(inputFile)).thenReturn(fileLinesContext);

    HashMap<InputFile, Set<Integer>> linesOfCode = new HashMap<>();
    SquidAstVisitor<Grammar> visitor = new FileLinesVisitor(fileLinesContextFactory, fileSystem, linesOfCode);

    PythonAstScanner.scanSingleFile(file.getPath(), visitor);

    assertThat(linesOfCode).hasSize(1);
    assertThat(linesOfCode.get(inputFile)).as("Lines of codes").containsOnly(2, 4, 7, 8, 9, 10, 11, 12, 14, 15, 17, 21);

    verifyInvocation(fileLinesContext, CoreMetrics.NCLOC_DATA_KEY, 2, 4, 7, 8, 9, 10, 11, 12, 14, 15, 17, 21);
    verifyInvocation(fileLinesContext, CoreMetrics.COMMENT_LINES_DATA_KEY, 1, 4, 6, 11, 13, 14, 17, 18, 18, 19, 20);
    verify(fileLinesContext).save();
    verifyNoMoreInteractions(fileLinesContext);
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
