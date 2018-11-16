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

import com.google.common.collect.ImmutableMap;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.function.Predicate;
import org.junit.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.config.Configuration;
import org.sonar.api.config.internal.ConfigurationBridge;
import org.sonar.api.config.internal.MapSettings;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.TestUtils;

import static org.assertj.core.api.Assertions.assertThat;

public class PylintImportSensorTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/pylint");
  private final SensorContextTester context = SensorContextTester.create(baseDir);

  @Test
  public void parse_report() {
    context.settings().setProperty(PylintImportSensor.REPORT_PATH_KEY, "pylint-report.txt");

    File file = new File(baseDir, "src/file1.py");
    DefaultInputFile inputFile = TestInputFileBuilder.create("", "src/file1.py")
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(file, StandardCharsets.UTF_8))
      .build();
    context.fileSystem().add(inputFile);

    context.setActiveRules((new ActiveRulesBuilder())
      .create(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, "C0103"))
      .setName("Invalid name")
      .activate()
      .create(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, "C0111"))
      .setName("Missing docstring")
      .activate()
      .build());

    PylintImportSensor sensor = new PylintImportSensor(context.config());
    sensor.execute(context);
    assertThat(context.allIssues()).hasSize(3);
    assertThat(context.allIssues()).extracting(issue -> issue.primaryLocation().inputComponent().key())
      .containsOnly(inputFile.key());
  }

  @Test
  public void sensor_descriptor() {
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    new PylintImportSensor(context.config()).describe(descriptor);
    assertThat(descriptor.name()).isEqualTo("PylintImportSensor");
    assertThat(descriptor.languages()).containsOnly("py");
    assertThat(descriptor.type()).isEqualTo(InputFile.Type.MAIN);
    assertThat(descriptor.ruleRepositories()).containsExactly(PylintRuleRepository.REPOSITORY_KEY);
    Predicate<Configuration> configurationPredicate = descriptor.configurationPredicate();
    assertThat(configurationPredicate.test(configuration(ImmutableMap.of(PylintImportSensor.REPORT_PATH_KEY, "something")))).isTrue();
    assertThat(configurationPredicate.test(configuration(ImmutableMap.of("xxx", "yyy")))).isFalse();
  }

  private Configuration configuration(Map<String, String> mapproperties) {
    return new ConfigurationBridge(new MapSettings().addProperties(mapproperties));
  }

}
