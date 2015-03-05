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

import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.config.Settings;
import org.sonar.api.profiles.RulesProfile;
import org.sonar.api.resources.Project;
import org.sonar.api.rules.ActiveRule;
import org.sonar.api.rules.RuleFinder;
import org.sonar.plugins.python.Python;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class PylintSensorTest {
  private RuleFinder ruleFinder;
  private PylintConfiguration conf;
  private RulesProfile profile;

  @Before
  public void init() {
    ruleFinder = mock(RuleFinder.class);
    conf = mock(PylintConfiguration.class);
    profile = mock(RulesProfile.class);
  }

  @Test
  public void shouldNotThrowWhenInstantiating() {
    DefaultFileSystem fileSystem = new DefaultFileSystem();
    new PylintSensor(ruleFinder, conf, profile, fileSystem, mock(ResourcePerspectives.class), new Settings());
  }

  @Test
  public void shouldExecuteOnlyWhenNecessary() {
    // which means: only on python projects and only if
    // there is at least one active pylint rule

    DefaultFileSystem fileSystemPython = new DefaultFileSystem();
    DefaultFileSystem fileSystemForeign = new DefaultFileSystem();
    DefaultInputFile inputFile1 = new DefaultInputFile("src/test/resources/example_project/example.py").setLanguage(Python.KEY);
    inputFile1.setAbsolutePath((new File("src/test/resources/example_project/example.py")).getAbsolutePath());
    fileSystemPython.add(inputFile1);

    Project pythonProject = mock(Project.class);
    Project foreignProject = mock(Project.class);
    RulesProfile emptyProfile = mock(RulesProfile.class);
    RulesProfile pylintProfile = createPylintProfile();

    checkNecessityOfExecution(pythonProject, pylintProfile, fileSystemPython, true);
    checkNecessityOfExecution(pythonProject, emptyProfile, fileSystemPython, false);

    checkNecessityOfExecution(foreignProject, pylintProfile, fileSystemForeign, false);
    checkNecessityOfExecution(foreignProject, emptyProfile, fileSystemForeign, false);
  }

  private void checkNecessityOfExecution(Project project, RulesProfile profile, DefaultFileSystem fileSystem, boolean shouldExecute) {
    PylintSensor sensor = new PylintSensor(ruleFinder, conf, profile, fileSystem, mock(ResourcePerspectives.class), new Settings());
    assertThat(sensor.shouldExecuteOnProject(project)).isEqualTo(shouldExecute);
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
