/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.caching;

import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.SonarProduct;
import org.sonar.api.SonarRuntime;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.config.Configuration;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.api.utils.Version;
import org.sonar.plugins.python.api.caching.CacheContext;

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

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
    assertThat(cacheContext.isCacheEnabled()).isTrue();
  }

  @Test
  void cache_context_of_disabled_cache() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITH_CACHING, false);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
  }

  @Test
  void cache_context_on_sonarlint() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARLINT, VERSION_WITH_CACHING, true);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
  }

  @Test
  void cache_context_on_old_version() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITHOUT_CACHING, true);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
  }

  @Test
  void cache_context_with_sonar_modules_property() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITH_CACHING, true);
    Configuration configuration = mock(Configuration.class);
    when(configuration.get("sonar.modules")).thenReturn(Optional.of("module1, module2"));
    when(sensorContext.config()).thenReturn(configuration);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
    assertThat(logTester.logs(Level.INFO)).contains(EXPECTED_SONAR_MODULE_LOG);
  }

  @Test
  void cache_context_when_cache_disabled_no_sonar_module_logs() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITH_CACHING, false);
    Configuration configuration = mock(Configuration.class);
    when(configuration.get("sonar.modules")).thenReturn(Optional.of("module1, module2"));
    when(sensorContext.config()).thenReturn(configuration);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
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
}
