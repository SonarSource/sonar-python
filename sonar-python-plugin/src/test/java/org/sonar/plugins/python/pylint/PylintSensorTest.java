/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import java.util.Optional;
import javax.annotation.Nullable;
import org.junit.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.config.Configuration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

public class PylintSensorTest {

  private PylintConfiguration conf = mock(PylintConfiguration.class);

  @Test
  public void sensor_descriptor() {
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    new PylintSensor(conf, new SimpleConfig(), new PylintRuleRepository()).describe(descriptor);
    assertThat(descriptor.name()).isEqualTo("PylintSensor");
    assertThat(descriptor.languages()).containsOnly("py");
    assertThat(descriptor.type()).isEqualTo(InputFile.Type.MAIN);
    assertThat(descriptor.ruleRepositories()).containsExactly(PylintRuleRepository.REPOSITORY_KEY);
  }

  @Test
  public void shouldExecuteOnlyWhenNecessary() {
    assertThat(shouldExecute(null)).isTrue();
    assertThat(shouldExecute("result.txt")).isFalse();
  }

  private boolean shouldExecute(@Nullable String pylintReportPath) {
    Configuration settings = new SimpleConfig(PylintImportSensor.REPORT_PATH_KEY, pylintReportPath);
    PylintSensor sensor = new PylintSensor(conf, settings, new PylintRuleRepository());
    return sensor.shouldExecute();
  }

  private class SimpleConfig implements Configuration {

    private final String key;
    private final String value;

    private SimpleConfig() {
      this(null, null);
    }

    private SimpleConfig(@Nullable String key, @Nullable String value) {
      this.key = key;
      this.value = value;
    }

    @Override
    public Optional<String> get(String key) {
      return Optional.ofNullable(key.equals(this.key) ? value : null);
    }

    @Override
    public boolean hasKey(String key) {
      return key.equals(this.key);
    }

    @Override
    public String[] getStringArray(String key) {
      throw new UnsupportedOperationException();
    }
  }

}
