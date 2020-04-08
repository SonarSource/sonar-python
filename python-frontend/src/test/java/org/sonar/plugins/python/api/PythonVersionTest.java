/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.plugins.python.api;

import org.junit.Test;
import org.sonar.api.utils.log.LogTester;
import org.sonar.api.utils.log.LoggerLevel;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonVersionTest {

  @org.junit.Rule
  public LogTester logTester = new LogTester();

  @Test
  public void python2_only() {
    assertThat(PythonVersion.fromString("<= 2.7").isPython2Only()).isTrue();
    assertThat(PythonVersion.fromString("< 3").isPython2Only()).isTrue();
    assertThat(PythonVersion.fromString(">= 3").isPython2Only()).isFalse();
    assertThat(PythonVersion.fromString(">= 2").isPython2Only()).isFalse();
    assertThat(PythonVersion.fromString(">= 3.6, < 3.7").isPython2Only()).isFalse();
    assertThat(PythonVersion.fromString("< 1").isPython2Only()).isFalse();
    assertThat(PythonVersion.fromString("> 3.2").isPython2Only()).isFalse();
    assertThat(PythonVersion.fromString("< 2.7.1").isPython2Only()).isTrue();
  }

  @Test
  public void python3_only() {
    assertThat(PythonVersion.fromString("<= 2.7").isPython3Only()).isFalse();
    assertThat(PythonVersion.fromString("< 3").isPython3Only()).isFalse();
    assertThat(PythonVersion.fromString(">= 2").isPython3Only()).isFalse();
    assertThat(PythonVersion.fromString(">= 3").isPython3Only()).isTrue();
    assertThat(PythonVersion.fromString(">= 3.6").isPython3Only()).isTrue();
    assertThat(PythonVersion.fromString(">= 3.6, < 3.7").isPython3Only()).isTrue();
    assertThat(PythonVersion.fromString("< 1").isPython3Only()).isFalse();
    assertThat(PythonVersion.fromString("> 3.2").isPython3Only()).isTrue();
  }

  @Test
  public void version_out_of_range() {
    PythonVersion pythonVersion = PythonVersion.fromString("> 4");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();
    assertThat(logTester.logs(LoggerLevel.WARN)).contains("Python version range '> 4' is not supported. Versions must be between 2.5 and 3.8.");

    pythonVersion = PythonVersion.fromString("< 5");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();
    assertThat(logTester.logs(LoggerLevel.WARN)).contains("Python version range '< 5' is not supported. Versions must be between 2.5 and 3.8.");
  }

  @Test
  public void disjoint_intervals() {
    PythonVersion pythonVersion = PythonVersion.fromString("< 2.9, > 3.5");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();

    pythonVersion = PythonVersion.fromString("> 3.5, < 2.9");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();
  }

  @Test
  public void overlapping_intervals() {
    PythonVersion pythonVersion = PythonVersion.fromString("> 3.1, > 3.5");
    assertThat(pythonVersion.isPython3Only()).isTrue();
    assertThat(pythonVersion.isPython2Only()).isFalse();

    pythonVersion = PythonVersion.fromString("> 3.5, > 3.1");
    assertThat(pythonVersion.isPython3Only()).isTrue();
    assertThat(pythonVersion.isPython2Only()).isFalse();

    pythonVersion = PythonVersion.fromString("< 2.6, < 2.9");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isTrue();
  }

  @Test
  public void more_than_two_intervals() {
    PythonVersion pythonVersion = PythonVersion.fromString("> 3, < 2.7, > 3.8");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();
    assertThat(logTester.logs(LoggerLevel.WARN)).contains("Error while parsing value of parameter 'sonar.python.version' (> 3, < 2.7, > 3.8). Only two intervals are supported (e.g. >= 3.6, < 3.8).");
  }

  @Test
  public void error_while_parsing_version() {
    PythonVersion pythonVersion = PythonVersion.fromString("> foo");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();
    assertThat(logTester.logs(LoggerLevel.WARN)).contains("Error while parsing value of parameter 'sonar.python.version' (> foo). Versions must be between 2.5 and 3.8.");

    pythonVersion = PythonVersion.fromString("< bar");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();
    assertThat(logTester.logs(LoggerLevel.WARN)).contains("Error while parsing value of parameter 'sonar.python.version' (< bar). Versions must be between 2.5 and 3.8.");
  }

  @Test
  public void error_while_parsing_version_using_equal() {
    PythonVersion pythonVersion = PythonVersion.fromString("= 3.4");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();
    assertThat(logTester.logs(LoggerLevel.WARN)).contains("Error while parsing value of parameter 'sonar.python.version' (= 3.4). Only intervals are supported (e.g. >= 3.6, < 3.8).");
  }

  @Test
  public void allVersions() {
    assertThat(PythonVersion.allVersions().isPython3Only()).isFalse();
    assertThat(PythonVersion.allVersions().isPython2Only()).isFalse();

    PythonVersion pythonVersion = PythonVersion.fromString("< 3.9");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();

    pythonVersion = PythonVersion.fromString(">= 2.5");
    assertThat(pythonVersion.isPython3Only()).isFalse();
    assertThat(pythonVersion.isPython2Only()).isFalse();

    assertThat(logTester.logs(LoggerLevel.WARN)).isEmpty();
  }
}
