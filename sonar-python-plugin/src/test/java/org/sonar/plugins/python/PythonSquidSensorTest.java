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
package org.sonar.plugins.python;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.issue.Issuable;
import org.sonar.api.issue.Issue;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.resources.Project;
import org.sonar.api.rule.RuleKey;
import org.sonar.python.checks.CheckList;

import java.io.File;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;


public class PythonSquidSensorTest {

  private PythonSquidSensor sensor;
  private DefaultFileSystem fs = new DefaultFileSystem();
  ResourcePerspectives perspectives;

  @Before
  public void setUp() {
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(InputFile.class))).thenReturn(fileLinesContext);
    ActiveRules activeRules = (new ActiveRulesBuilder())
        .create(RuleKey.of(CheckList.REPOSITORY_KEY, "PrintStatementUsage"))
        .setName("Print Statement Usage")
        .activate()
        .build();
    CheckFactory checkFactory = new CheckFactory(activeRules);
    perspectives = mock(ResourcePerspectives.class);
    sensor = new PythonSquidSensor(fileLinesContextFactory, fs, perspectives, checkFactory);
  }

  @Test
  public void should_execute_on_python_project() {
    Project project = mock(Project.class);
    assertThat(sensor.toString()).isEqualTo("PythonSquidSensor");
    assertThat(sensor.shouldExecuteOnProject(project)).isFalse();
    fs.add(new DefaultInputFile("test.py").setLanguage(Python.KEY));
    assertThat(sensor.shouldExecuteOnProject(project)).isTrue();
  }

  @Test
  public void should_analyse() {
    String relativePath = "src/test/resources/org/sonar/plugins/python/code_chunks_2.py";
    DefaultInputFile inputFile = new DefaultInputFile(relativePath).setLanguage(Python.KEY);
    inputFile.setAbsolutePath((new File(relativePath)).getAbsolutePath());
    fs.add(inputFile);

    Issuable issuable = mock(Issuable.class);
    Issuable.IssueBuilder issueBuilder = mock(Issuable.IssueBuilder.class);
    when(perspectives.as(Mockito.eq(Issuable.class), Mockito.any(InputFile.class))).thenReturn(issuable);
    when(issuable.newIssueBuilder()).thenReturn(issueBuilder);
    when(issueBuilder.ruleKey(Mockito.any(RuleKey.class))).thenReturn(issueBuilder);
    when(issueBuilder.line(Mockito.any(Integer.class))).thenReturn(issueBuilder);
    when(issueBuilder.message(Mockito.any(String.class))).thenReturn(issueBuilder);

    Project project = new Project("key");
    SensorContext context = mock(SensorContext.class);
    sensor.analyse(project, context);

    verify(context).saveMeasure(Mockito.any(InputFile.class), Mockito.eq(CoreMetrics.FILES), Mockito.eq(1.0));
    verify(context).saveMeasure(Mockito.any(InputFile.class), Mockito.eq(CoreMetrics.LINES), Mockito.eq(29.0));
    verify(context).saveMeasure(Mockito.any(InputFile.class), Mockito.eq(CoreMetrics.NCLOC), Mockito.eq(25.0));
    verify(context).saveMeasure(Mockito.any(InputFile.class), Mockito.eq(CoreMetrics.STATEMENTS), Mockito.eq(23.0));
    verify(context).saveMeasure(Mockito.any(InputFile.class), Mockito.eq(CoreMetrics.FUNCTIONS), Mockito.eq(4.0));
    verify(context).saveMeasure(Mockito.any(InputFile.class), Mockito.eq(CoreMetrics.CLASSES), Mockito.eq(1.0));
    verify(context).saveMeasure(Mockito.any(InputFile.class), Mockito.eq(CoreMetrics.COMPLEXITY), Mockito.eq(4.0));
    verify(context).saveMeasure(Mockito.any(InputFile.class), Mockito.eq(CoreMetrics.COMMENT_LINES), Mockito.eq(9.0));
    verify(issuable).addIssue(Mockito.any(Issue.class));

  }

}
