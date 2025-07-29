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
package org.sonar.plugins.python.indexer;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import org.sonar.python.project.config.ProjectConfigurationBuilder;

class PythonIndexerWrapperTest {

  @Test
  void testEmptyConstructor() {
    PythonIndexerWrapper wrapper = new PythonIndexerWrapper();
    assertThat(wrapper.indexer()).isNull();
  }

  @Test
  void testConstructorWithParameter() {
    TestModuleFileSystem moduleFileSystem = new TestModuleFileSystem(new ArrayList<>());
    PythonIndexerWrapper wrapper = new PythonIndexerWrapper(new SonarLintPythonIndexer(moduleFileSystem, new ProjectConfigurationBuilder()));
    assertThat(wrapper.indexer()).isNotNull().isInstanceOf(PythonIndexer.class);
  }

}
