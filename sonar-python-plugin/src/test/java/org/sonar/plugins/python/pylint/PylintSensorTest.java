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
package org.sonar.plugins.python.pylint;

import com.google.common.collect.ImmutableList;
import org.apache.commons.collections.ListUtils;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.profiles.RulesProfile;
import org.sonar.api.resources.Project;
import org.sonar.api.rules.ActiveRule;
import org.sonar.api.rules.RuleFinder;
import org.sonar.api.scan.filesystem.FileQuery;
import org.sonar.api.scan.filesystem.ModuleFileSystem;
import org.sonar.plugins.python.Python;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class PylintSensorTest {

  private ModuleFileSystem fs;
  private RuleFinder ruleFinder;
  private PylintConfiguration conf;
  private RulesProfile profile;

  @Before
  public void init() {
    ruleFinder = mock(RuleFinder.class);
    conf = mock(PylintConfiguration.class);
    profile = mock(RulesProfile.class);

    fs = mock(ModuleFileSystem.class);
  }

  @Test
  public void shouldntThrowWhenInstantiating() {
    new PylintSensor(ruleFinder, conf, profile, fs, mock(ResourcePerspectives.class));
  }

  @Test
  public void shouldExecuteOnlyWhenNecessary() {
    // which means: only on python projects and only if
    // there is at least one active pylint rule

    Project pythonProject = createProjectForLanguage(Python.KEY);
    Project foreignProject = createProjectForLanguage("whatever");
    RulesProfile emptyProfile = mock(RulesProfile.class);
    RulesProfile pylintProfile = createPylintProfile();

    when(fs.files(any(FileQuery.class))).thenReturn(ImmutableList.of(new File("/tmp")));
    checkNecessityOfExecution(pythonProject, pylintProfile, true);
    checkNecessityOfExecution(pythonProject, emptyProfile, false);

    when(fs.files(any(FileQuery.class))).thenReturn(ListUtils.EMPTY_LIST);
    checkNecessityOfExecution(foreignProject, pylintProfile, false);
    checkNecessityOfExecution(foreignProject, emptyProfile, false);
  }

  @Test
  public void analyse() {


  }

  private void checkNecessityOfExecution(Project project, RulesProfile profile, boolean shouldExecute) {
    PylintSensor sensor = new PylintSensor(ruleFinder, conf, profile, fs, mock(ResourcePerspectives.class));
    assertThat(sensor.shouldExecuteOnProject(project)).isEqualTo(shouldExecute);
  }

  private static Project createProjectForLanguage(String languageKey) {
    Project project = mock(Project.class);
    when(project.getLanguageKey()).thenReturn(languageKey);
    return project;
  }

  private static RulesProfile createPylintProfile() {
    List<ActiveRule> rules = new LinkedList<ActiveRule>();
    rules.add(mock(ActiveRule.class));

    RulesProfile profile = mock(RulesProfile.class);
    when(profile.getActiveRulesByRepository(PylintRuleRepository.REPOSITORY_KEY))
      .thenReturn(rules);

    return profile;
  }

}
