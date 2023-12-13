/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.caching;

import java.io.IOException;
import java.util.Optional;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.SonarProduct;
import org.sonar.api.SonarRuntime;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.config.Configuration;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.api.utils.Version;
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CacheContextImplTest {

  private static final Version VERSION_WITH_CACHING = Version.create(9, 7);
  private static final Version VERSION_WITHOUT_CACHING = Version.create(9, 6);
  private static final String EXPECTED_SONAR_MODULE_LOG = "Caching will be disabled for this analysis due to the use of the \"sonar.modules\" property.";

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @Test
  void cache_context_of_enabled_cache() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITH_CACHING, true);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext, null);
    assertThat(cacheContext.isCacheEnabled()).isTrue();
  }

  @Test
  void cache_context_of_disabled_cache() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITH_CACHING, false);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext, null);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
  }

  @Test
  void caching_is_always_enabled_if_there_is_a_sonarlint_cache() {
    SensorContext sensorContext;
    CacheContext cacheContext;

    sensorContext = sensorContext(SonarProduct.SONARLINT, VERSION_WITH_CACHING, true);
    cacheContext = CacheContextImpl.of(sensorContext, new SonarLintCache());
    Assertions.assertThat(cacheContext.isCacheEnabled()).isTrue();

    sensorContext = sensorContext(SonarProduct.SONARLINT, VERSION_WITH_CACHING, false);
    cacheContext = CacheContextImpl.of(sensorContext, new SonarLintCache());
    Assertions.assertThat(cacheContext.isCacheEnabled()).isTrue();
  }

  @Test
  void cache_context_on_old_version() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITHOUT_CACHING, true);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext, null);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
  }

  @Test
  void cache_context_with_sonar_modules_property() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITH_CACHING, true);
    Configuration configuration = mock(Configuration.class);
    when(configuration.get("sonar.modules")).thenReturn(Optional.of("module1, module2"));
    when(sensorContext.config()).thenReturn(configuration);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext, null);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
    assertThat(logTester.logs(Level.INFO)).contains(EXPECTED_SONAR_MODULE_LOG);
  }

  @Test
  void cache_context_when_cache_disabled_no_sonar_module_logs() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITH_CACHING, false);
    Configuration configuration = mock(Configuration.class);
    when(configuration.get("sonar.modules")).thenReturn(Optional.of("module1, module2"));
    when(sensorContext.config()).thenReturn(configuration);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext, null);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
    assertThat(logTester.logs(Level.INFO)).doesNotContain(EXPECTED_SONAR_MODULE_LOG);
  }

  @Test
  void dummy_cache() {
    CacheContext dummyCache = CacheContextImpl.dummyCache();
    assertThat(dummyCache.isCacheEnabled()).isFalse();
  }

  private static SensorContext sensorContext(SonarProduct product, Version version, boolean isCacheEnabled) {
    SonarRuntime runtime = mock(SonarRuntime.class);
    when(runtime.getProduct()).thenReturn(product);
    when(runtime.getApiVersion()).thenReturn(version);

    SensorContext sensorContext = mock(SensorContext.class);
    when(sensorContext.runtime()).thenReturn(runtime);
    when(sensorContext.isCacheEnabled()).thenReturn(isCacheEnabled);
    Configuration configuration = mock(Configuration.class);
    when(sensorContext.config()).thenReturn(configuration);

    return sensorContext;
  }

  @Test
  void use_dummy_cache_in_sonarlint_context_if_there_is_no_provided_sonarlint_cache() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARLINT, VERSION_WITH_CACHING, true);

    var cacheContext = CacheContextImpl.of(sensorContext, null);
    Assertions.assertThat(cacheContext.isCacheEnabled()).isFalse();
    Assertions.assertThat(cacheContext.getWriteCache()).isInstanceOf(DummyCache.class);
    Assertions.assertThat(cacheContext.getReadCache()).isInstanceOf(DummyCache.class);
  }

  @Test
  void sonarlint_cache_is_used_when_provided() throws IOException {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARLINT, VERSION_WITH_CACHING, true);

    var sonarLintCache = new SonarLintCache();
    var cacheContext = CacheContextImpl.of(sensorContext, sonarLintCache);
    Assertions.assertThat(cacheContext.isCacheEnabled()).isTrue();
    Assertions.assertThat(cacheContext.getWriteCache()).isInstanceOf(PythonWriteCache.class);
    Assertions.assertThat(cacheContext.getReadCache()).isInstanceOf(PythonReadCache.class);

    byte[] bytes = {0};
    sonarLintCache.write("foo", bytes);
    PythonReadCache readCache = cacheContext.getReadCache();
    try (var inputStream = readCache.read("foo")) {
      Assertions.assertThat(inputStream.readAllBytes()).isEqualTo(bytes);
    }
  }
}
