/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.project.config.ProjectConfigurationBuilder;
import org.sonar.python.semantic.v2.typetable.CachedTypeTable;
import org.sonar.python.semantic.v2.typetable.TypeTable;

import static org.assertj.core.api.Assertions.assertThat;

class PythonIndexerTest {
  
  @Test
  void test_projectLevelTypeTableIsCached() {
    TestPythonIndexer pythonIndexer = new TestPythonIndexer(new ProjectConfigurationBuilder());
    TypeTable typeTable = pythonIndexer.projectLevelTypeTable();
    assertThat(typeTable).isInstanceOf(CachedTypeTable.class);
  }

  private static class TestPythonIndexer extends PythonIndexer {

    protected TestPythonIndexer(ProjectConfigurationBuilder projectConfigurationBuilder) {
      super(projectConfigurationBuilder);
    }

    @Override
    public void buildOnce(SensorContext context) {
      // do nothing
    }

    @Override
    public void postAnalysis(SensorContext context) {
      // do nothing
    }

    @Override
    public CacheContext cacheContext() {
      return CacheContextImpl.dummyCache();
    }
  }
}
