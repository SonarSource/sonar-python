/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python;

import java.io.IOException;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.config.Configuration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ScannerTest {

  private SensorContext context;
  private Configuration configuration;
  private Scanner scanner;

  private class TestScanner extends Scanner {

    protected TestScanner(SensorContext context) {
      super(context);
    }

    @Override
    protected void logStart(int numThreads) {
      throw new UnsupportedOperationException("Unimplemented method 'logStart'");
    }

    @Override
    protected String name() {
      throw new UnsupportedOperationException("Unimplemented method 'name'");
    }

    @Override
    protected void scanFile(PythonInputFile file) throws IOException {
      throw new UnsupportedOperationException("Unimplemented method 'scanFile'");
    }

    @Override
    protected void processException(Exception e, PythonInputFile file) {
      throw new UnsupportedOperationException("Unimplemented method 'processException'");
    }

  }

  @BeforeEach
  void setUp() {
    context = mock(SensorContext.class);
    configuration = mock(Configuration.class);
    when(context.config()).thenReturn(configuration);
    scanner = new TestScanner(context);
  }

  @Test
  void testGetNumberOfThreads_whenPropertySetToValidValue_shouldReturnConfiguredValue() {
    when(configuration.getInt("sonar.python.analysis.threads")).thenReturn(Optional.of(4));
    int numberOfThreads = scanner.getNumberOfThreads(context);
    assertThat(numberOfThreads).isEqualTo(4);
  }

  @ParameterizedTest
  @ValueSource(ints = {-1, 0, -12})
  void testGetNumberOfThreads_whenPropertySetIncorrectly_shouldReturnOne(int threads) {
    when(configuration.getInt("sonar.python.analysis.threads")).thenReturn(Optional.of(threads));

    int numberOfThreads = scanner.getNumberOfThreads(context);
    assertThat(numberOfThreads).isEqualTo(1);
  }

  @Test
  void testGetNumberOfThreads_whenParallelPropertyIsFalse_shouldReturnOne() {
    when(configuration.getInt("sonar.python.analysis.threads")).thenReturn(Optional.of(4));
    when(configuration.getBoolean("sonar.python.analysis.parallel")).thenReturn(Optional.of(false));

    int numberOfThreads = scanner.getNumberOfThreads(context);
    assertThat(numberOfThreads).isEqualTo(1);
  }

  @Test
  void testGetNumberOfThreads_whenParallelPropertyIsTrueOrNotSet_shouldReturnConfiguredValue() {
    when(configuration.getInt("sonar.python.analysis.threads")).thenReturn(Optional.of(4));
    when(configuration.getBoolean("sonar.python.analysis.parallel")).thenReturn(Optional.empty());
    int numberOfThreads = scanner.getNumberOfThreads(context);
    assertThat(numberOfThreads).isEqualTo(4);

    when(configuration.getBoolean("sonar.python.analysis.parallel")).thenReturn(Optional.of(true));
    numberOfThreads = scanner.getNumberOfThreads(context);
    assertThat(numberOfThreads).isEqualTo(4);

  }
}
