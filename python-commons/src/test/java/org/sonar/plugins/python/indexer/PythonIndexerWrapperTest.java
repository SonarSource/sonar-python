/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.api.config.Configuration;
import org.sonar.scanner.plugin.api.impl.config.MapSettings;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.project.config.ProjectConfigurationBuilder;

class PythonIndexerWrapperTest {

  private static final Configuration CONFIGURATION = new MapSettings().asConfig();

  @Test
  void testEmptyConstructor() {
    PythonIndexerWrapper wrapper = new PythonIndexerWrapper(CONFIGURATION);
    assertThat(wrapper.indexer()).isNull();
  }

  @Test
  void testConstructorWithParameter() {
    TestModuleFileSystem moduleFileSystem = new TestModuleFileSystem(new ArrayList<>());
    PythonIndexerWrapper wrapper = new PythonIndexerWrapper(CONFIGURATION,
      new SonarLintPythonIndexer(moduleFileSystem, new ProjectConfigurationBuilder()));
    assertThat(wrapper.indexer()).isNotNull().isInstanceOf(PythonIndexer.class);
  }

  @Test
  void indexerWhichIsNotApplicableIsDropped() {
    PythonIndexerWrapper wrapper = new PythonIndexerWrapper(CONFIGURATION, new NotApplicableIndexer());
    assertThat(wrapper.indexer()).isNull();
  }

  private static class NotApplicableIndexer extends PythonIndexer {

    NotApplicableIndexer() {
      super(new ProjectConfigurationBuilder());
    }

    @Override
    public boolean isApplicable(Configuration configuration) {
      return false;
    }

    @Override
    public void buildOnce(SensorContext context) {
      // no op
    }

    @Override
    public void postAnalysis(SensorContext context) {
      // no op
    }

    @Override
    public CacheContext cacheContext() {
      return CacheContextImpl.dummyCache();
    }
  }
}
