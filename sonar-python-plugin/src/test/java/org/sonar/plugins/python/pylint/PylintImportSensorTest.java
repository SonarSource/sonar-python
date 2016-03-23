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
package org.sonar.plugins.python.pylint;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.config.Settings;
import org.sonar.api.issue.*;
import org.sonar.api.resources.Project;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.Python;

import java.io.File;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class PylintImportSensorTest {
  private Settings settings;
  private DefaultFileSystem fileSystem;
  private ActiveRules activeRules;
  private SensorContext context;
  private InputFile inputFile;

  @Before
  public void init() {
    settings = new Settings();
    settings.setProperty(PylintImportSensor.REPORT_PATH_KEY, "pylint-report.txt");
    fileSystem = new DefaultFileSystem();
    fileSystem.setBaseDir(new File("src/test/resources/org/sonar/plugins/python/pylint"));
    inputFile = new DefaultInputFile("src/prod.py").setLanguage(Python.KEY);
    fileSystem.add(inputFile);
    activeRules = (new ActiveRulesBuilder())
        .create(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, "C0103"))
        .setName("Invalid name")
        .activate()
        .create(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, "C0111"))
        .setName("Missing docstring")
        .activate()
        .build();
    context = mock(SensorContext.class);
  }

  @Test
  public void shouldNotThrowWhenInstantiating() {
    new PylintImportSensor(settings, activeRules, fileSystem, mock(ResourcePerspectives.class));
  }

  @Test
  public void shouldExecuteOnlyWhenNecessary() {
    DefaultFileSystem fileSystemForeign = new DefaultFileSystem();

    Project project = mock(Project.class);

    ActiveRules emptyProfile = mock(ActiveRules.class);

    checkNecessityOfExecution(project, activeRules, fileSystem, true);
    checkNecessityOfExecution(project, emptyProfile, fileSystem, false);

    checkNecessityOfExecution(project, activeRules, fileSystemForeign, false);
    checkNecessityOfExecution(project, emptyProfile, fileSystemForeign, false);
  }

  @Test
  public void parse_report() {
    ResourcePerspectives perspectives = mock(ResourcePerspectives.class);
    Issuable issuable = mock(Issuable.class);
    when(perspectives.as(Issuable.class, inputFile)).thenReturn(issuable);
    Issuable.IssueBuilder issueBuilder = mock(Issuable.IssueBuilder.class);
    when(issuable.newIssueBuilder()).thenReturn(issueBuilder);
    when(issueBuilder.ruleKey(Mockito.any(RuleKey.class))).thenReturn(issueBuilder);
    when(issueBuilder.line(Mockito.any(Integer.class))).thenReturn(issueBuilder);
    when(issueBuilder.message(Mockito.any(String.class))).thenReturn(issueBuilder);

    PylintImportSensor sensor = new PylintImportSensor(settings, activeRules, fileSystem, perspectives);
    sensor.analyse(mock(Project.class), context);

    verify(issuable, times(3)).addIssue(any(org.sonar.api.issue.Issue.class));

  }


  private void checkNecessityOfExecution(Project project, ActiveRules currentActiveRules, DefaultFileSystem fileSystem, boolean shouldExecute) {
    PylintImportSensor sensor = new PylintImportSensor(settings, currentActiveRules, fileSystem, mock(ResourcePerspectives.class));
    assertThat(sensor.shouldExecuteOnProject(project)).isEqualTo(shouldExecute);
  }

}
