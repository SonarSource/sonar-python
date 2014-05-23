/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python;

import com.google.common.collect.ImmutableList;
import com.sonar.sslr.api.Grammar;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.ProjectFileSystem;
import org.sonar.python.metrics.FileLinesVisitor;
import org.sonar.squidbridge.SquidAstVisitor;
import org.sonar.squidbridge.api.SourceFile;

import java.io.File;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

public class FileLinesVisitorTest {

  private static final File BASE_DIR = new File("src/test/resources/metrics");

  @Mock
  private Project project;
  @Mock
  private FileLinesContextFactory fileLinesContextFactory;
  @Mock
  private ProjectFileSystem fileSystem;

  @Before
  public void before() {
    MockitoAnnotations.initMocks(this);
    when(project.getFileSystem()).thenReturn(fileSystem);
    when(fileSystem.getSourceDirs()).thenReturn(ImmutableList.of(BASE_DIR));
  }

  @Test
  public void test() {
    File file = new File(BASE_DIR, "file_lines.py");
    org.sonar.api.resources.File resource = org.sonar.api.resources.File.fromIOFile(file, project);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(resource)).thenReturn(fileLinesContext);

    SquidAstVisitor<Grammar> visitor = new FileLinesVisitor(project, fileLinesContextFactory);

    SourceFile sourceFile = PythonAstScanner.scanSingleFile(file, visitor);
    verify(fileLinesContext).setIntValue(CoreMetrics.NCLOC_DATA_KEY, 1, 0);
    verify(fileLinesContext).setIntValue(CoreMetrics.NCLOC_DATA_KEY, 2, 1);
    verify(fileLinesContext).setIntValue(CoreMetrics.NCLOC_DATA_KEY, 3, 0);
    verify(fileLinesContext).setIntValue(CoreMetrics.NCLOC_DATA_KEY, 4, 1);
    verify(fileLinesContext).setIntValue(CoreMetrics.NCLOC_DATA_KEY, 5, 0);
    verify(fileLinesContext).setIntValue(CoreMetrics.COMMENT_LINES_DATA_KEY, 1, 1);
    verify(fileLinesContext).setIntValue(CoreMetrics.COMMENT_LINES_DATA_KEY, 2, 0);
    verify(fileLinesContext).setIntValue(CoreMetrics.COMMENT_LINES_DATA_KEY, 3, 0);
    verify(fileLinesContext).setIntValue(CoreMetrics.COMMENT_LINES_DATA_KEY, 4, 1);
    verify(fileLinesContext).setIntValue(CoreMetrics.COMMENT_LINES_DATA_KEY, 5, 0);
    verify(fileLinesContext).save();
    verifyNoMoreInteractions(fileLinesContext);
  }

}
