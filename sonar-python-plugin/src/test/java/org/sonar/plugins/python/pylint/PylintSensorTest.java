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
package org.sonar.plugins.python.pylint;

import java.io.File;
import java.nio.file.Paths;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.config.MapSettings;
import org.sonar.api.resources.Project;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.Python;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

public class PylintSensorTest {
  private PylintConfiguration conf;
  private ActiveRules activeRules;
  private DefaultFileSystem fileSystem;

  @Before
  public void init() {
    conf = mock(PylintConfiguration.class);
    activeRules = (new ActiveRulesBuilder())
        .create(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, "C0103"))
        .setName("Invalid name")
        .activate()
        .build();
    fileSystem = new DefaultFileSystem(new File("src/test/resources/org/sonar/plugins/python/pylint"));
    fileSystem.setWorkDir(new File("target/"));
    DefaultInputFile inputFile = new DefaultInputFile("", "src/test/resources/example_project/example.py").setLanguage(Python.KEY);
    inputFile.setModuleBaseDir(Paths.get("").toAbsolutePath());
    fileSystem.add(inputFile);
  }

  @Test
  public void shouldExecuteOnlyWhenNecessary() {
    DefaultFileSystem fileSystemForeign = new DefaultFileSystem(Paths.get(""));
    Project project = mock(Project.class);

    checkNecessityOfExecution(project, activeRules, fileSystem, true);
    ActiveRules emptyActiveRules = (new ActiveRulesBuilder()).build();
    checkNecessityOfExecution(project, emptyActiveRules, fileSystem, false);

    checkNecessityOfExecution(project, activeRules, fileSystemForeign, false);
    checkNecessityOfExecution(project, emptyActiveRules, fileSystemForeign, false);
  }

  private void checkNecessityOfExecution(Project project, ActiveRules currentActiveRules, DefaultFileSystem currentFileSystem, boolean shouldExecute) {
    PylintSensor sensor = new PylintSensor(conf, currentActiveRules, currentFileSystem, mock(ResourcePerspectives.class), new MapSettings());
    assertThat(sensor.shouldExecuteOnProject(project)).isEqualTo(shouldExecute);
  }

}
