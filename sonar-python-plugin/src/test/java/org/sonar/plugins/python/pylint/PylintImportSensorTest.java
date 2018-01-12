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
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.FileMetadata;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.config.MapSettings;
import org.sonar.api.config.Settings;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.Python;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

public class PylintImportSensorTest {

  private File baseDir = new File("src/test/resources/org/sonar/plugins/python/pylint");
  private Settings settings;
  private DefaultFileSystem fileSystem;
  private ActiveRules activeRules;
  private DefaultInputFile inputFile;

  @Before
  public void init() {
    settings = new MapSettings();
    settings.setProperty(PylintImportSensor.REPORT_PATH_KEY, "pylint-report.txt");

    fileSystem = new DefaultFileSystem(baseDir);

    File file = new File(baseDir, "src/file1.py");
    inputFile = new DefaultInputFile("", "src/file1.py")
      .setLanguage(Python.KEY)
      .initMetadata(new FileMetadata().readMetadata(file, StandardCharsets.UTF_8));
    fileSystem.add(inputFile);
    activeRules = (new ActiveRulesBuilder())
        .create(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, "C0103"))
        .setName("Invalid name")
        .activate()
        .create(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, "C0111"))
        .setName("Missing docstring")
        .activate()
        .build();
  }

  @Test
  public void shouldExecuteOnlyWhenNecessary() {
    ActiveRules emptyProfile = mock(ActiveRules.class);
    Settings settingsWithoutProperty = new MapSettings();

    assertThat(shouldExecute(activeRules, settings)).isTrue();
    assertThat(shouldExecute(emptyProfile, settings)).isFalse();

    assertThat(shouldExecute(activeRules, settingsWithoutProperty)).isFalse();
    assertThat(shouldExecute(emptyProfile, settingsWithoutProperty)).isFalse();
  }

  private boolean shouldExecute(ActiveRules activeRules, Settings settings) {
    SensorContextTester ctx = SensorContextTester.create(baseDir);
    ctx.setActiveRules(activeRules);
    ctx.setSettings(settings);
    ctx.setFileSystem(fileSystem);
    AtomicBoolean executed = new AtomicBoolean(false);
    PylintImportSensor sensor = new PylintImportSensor(settings) {
      @Override
      protected void processReports(org.sonar.api.batch.sensor.SensorContext context, List<File> reports) {
        super.processReports(context, reports);
        executed.set(true);
      }
    };
    sensor.execute(ctx);
    return executed.get();
  }

  @Test
  public void parse_report() {
    SensorContextTester context = SensorContextTester.create(baseDir);
    context.setActiveRules(activeRules);
    context.setFileSystem(fileSystem);

    PylintImportSensor sensor = new PylintImportSensor(settings);
    sensor.execute(context);
    assertThat(context.allIssues()).hasSize(3);
    assertThat(context.allIssues()).extracting(issue -> issue.primaryLocation().inputComponent().key())
      .containsOnly(inputFile.key());
  }

}
