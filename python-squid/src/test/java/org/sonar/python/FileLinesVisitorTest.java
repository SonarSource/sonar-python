/*
 * Copyright (C) 2014 SonarSource SA
 * All rights reserved
 * mailto:contact AT sonarsource DOT com
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
