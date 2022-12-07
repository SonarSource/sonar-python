/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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

import org.junit.Ignore;
import org.junit.Test;
import org.sonar.api.SonarProduct;
import org.sonar.api.SonarRuntime;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.utils.Version;
import org.sonar.plugins.python.api.caching.CacheContext;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class CacheContextImplTest {

  private static final Version VERSION_WITH_CACHING = Version.create(9, 7);
  private static final Version VERSION_WITHOUT_CACHING = Version.create(9, 6);

  @Test
  @Ignore
  public void cache_context_of_enabled_cache() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITH_CACHING, true);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
    assertThat(cacheContext.isCacheEnabled()).isTrue();
  }

  @Test
  public void cache_context_of_disabled_cache() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITH_CACHING, false);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
  }

  @Test
  public void cache_context_on_sonarlint() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARLINT, VERSION_WITH_CACHING, true);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
  }

  @Test
  public void cache_context_on_old_version() {
    SensorContext sensorContext = sensorContext(SonarProduct.SONARQUBE, VERSION_WITHOUT_CACHING, true);

    CacheContext cacheContext = CacheContextImpl.of(sensorContext);
    assertThat(cacheContext.isCacheEnabled()).isFalse();
  }

  @Test
  public void dummy_cache() {
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

    return sensorContext;
  }
}
