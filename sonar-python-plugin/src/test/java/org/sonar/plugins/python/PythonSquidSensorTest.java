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

import java.io.File;
import java.util.Iterator;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.FileMetadata;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.batch.sensor.issue.Issue;
import org.sonar.api.batch.sensor.issue.IssueLocation;
import org.sonar.api.internal.google.common.base.Charsets;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.rule.RuleKey;
import org.sonar.python.checks.CheckList;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class PythonSquidSensorTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/squid-sensor");
  private SensorContextTester context = SensorContextTester.create(baseDir);
  private ActiveRules activeRules;

  @Test
  public void sensor_descriptor() {
    activeRules = (new ActiveRulesBuilder()).build();
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    sensor().describe(descriptor);

    assertThat(descriptor.name()).isEqualTo("Python Squid Sensor");
    assertThat(descriptor.languages()).containsOnly("py");
    assertThat(descriptor.type()).isEqualTo(Type.MAIN);
  }

  @Test
  public void test_execute() {
    activeRules = (new ActiveRulesBuilder())
      .create(RuleKey.of(CheckList.REPOSITORY_KEY, "PrintStatementUsage"))
      .setName("Print Statement Usage")
      .activate()
      .build();

    inputFile("file1.py");

    sensor().execute(context);

    String key = "moduleKey:file1.py";
    assertThat(context.measure(key, CoreMetrics.FILES).value()).isEqualTo(1);
    assertThat(context.measure(key, CoreMetrics.LINES).value()).isEqualTo(29);
    assertThat(context.measure(key, CoreMetrics.NCLOC).value()).isEqualTo(25);
    assertThat(context.measure(key, CoreMetrics.STATEMENTS).value()).isEqualTo(23);
    assertThat(context.measure(key, CoreMetrics.FUNCTIONS).value()).isEqualTo(4);
    assertThat(context.measure(key, CoreMetrics.CLASSES).value()).isEqualTo(1);
    assertThat(context.measure(key, CoreMetrics.COMPLEXITY).value()).isEqualTo(4);
    assertThat(context.measure(key, CoreMetrics.COMMENT_LINES).value()).isEqualTo(9);

    assertThat(context.allIssues()).hasSize(1);
    
    String msg = "number of TypeOfTypes for the highlighting of keyword 'def'";
    assertThat(context.highlightingTypeAt(key, 18, 2)).as(msg).hasSize(1);
  }

  @Test
  public void test_issues() {
    activeRules = (new ActiveRulesBuilder())
      .create(RuleKey.of(CheckList.REPOSITORY_KEY, "PrintStatementUsage"))
      .setName("Print Statement Usage")
      .activate()
      .create(RuleKey.of(CheckList.REPOSITORY_KEY, "S134"))
      .activate()
      .build();

    InputFile inputFile = inputFile("file2.py");
    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(2);
    Iterator<Issue> issuesIterator = context.allIssues().iterator();

    int checkedIssues = 0;

    while (issuesIterator.hasNext()) {
      Issue issue = issuesIterator.next();
      IssueLocation issueLocation = issue.primaryLocation();
      assertThat(issueLocation.inputComponent()).isEqualTo(inputFile);
      assertThat(issue.gap()).isNull();

      if (issue.ruleKey().rule().equals("S134")) {
        assertThat(issueLocation.message()).isEqualTo("Refactor this code to not nest more than 4 \"if\", \"for\", \"while\", \"try\" and \"with\" statements.");
        assertThat(issueLocation.textRange()).isEqualTo(inputFile.newRange(7, 16, 7, 18));
        assertThat(issue.flows()).hasSize(4);
        checkedIssues++;
      }

      if (issue.ruleKey().rule().equals("PrintStatementUsage")) {
        assertThat(issueLocation.message()).isEqualTo("Replace print statement by built-in function.");
        assertThat(issueLocation.textRange()).isEqualTo(inputFile.newRange(1, 0, 1, 50));
        assertThat(issue.flows()).isEmpty();
        checkedIssues++;
      }
    }

    assertThat(checkedIssues).isEqualTo(2);
  }

  private PythonSquidSensor sensor() {
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(InputFile.class))).thenReturn(fileLinesContext);
    CheckFactory checkFactory = new CheckFactory(activeRules);
    return new PythonSquidSensor(fileLinesContextFactory, checkFactory);
  }

  private InputFile inputFile(String name) {
    DefaultInputFile inputFile = new DefaultInputFile("moduleKey", name)
      .setModuleBaseDir(baseDir.toPath())
      .setType(Type.MAIN)
      .setLanguage(Python.KEY);
    context.fileSystem().add(inputFile);
    inputFile.initMetadata(new FileMetadata().readMetadata(inputFile.file(), Charsets.UTF_8));
    return inputFile;
  }

}
