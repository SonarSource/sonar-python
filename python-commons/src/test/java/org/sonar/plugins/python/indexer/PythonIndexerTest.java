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

import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
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

  @Test
  void concurrent_first_access_returns_one_non_null_table() throws Exception {
    var indexer = new TestPythonIndexer(new ProjectConfigurationBuilder());
    var start = new CountDownLatch(1);
    try (var executor = Executors.newFixedThreadPool(8)) {
      var calls = new ArrayList<Future<TypeTable>>();
      for (int i = 0; i < 8; i++) {
        calls.add(executor.submit(() -> {
          start.await();
          return indexer.projectLevelTypeTable();
        }));
      }
      start.countDown();
      var expected = calls.getFirst().get(10, TimeUnit.SECONDS);
      assertThat(expected).isNotNull();
      for (var call : calls) {
        assertThat(call.get(10, TimeUnit.SECONDS)).isSameAs(expected);
      }
      indexer.recreateProjectLevelTypeTable();
      assertThat(indexer.projectLevelTypeTable()).isNotNull().isNotSameAs(expected);
    }
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
