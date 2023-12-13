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

import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.SonarProduct;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.utils.Version;
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;

public class CacheContextImpl implements CacheContext {

  private static final Logger LOG = LoggerFactory.getLogger(CacheContextImpl.class);
  private static final Version MINIMUM_RUNTIME_VERSION = Version.create(9, 7);

  private final boolean isCacheEnabled;
  private final PythonWriteCache writeCache;
  private final PythonReadCache readCache;

  public CacheContextImpl(boolean isCacheEnabled, PythonWriteCache writeCache, PythonReadCache readCache) {
    this.isCacheEnabled = isCacheEnabled;
    this.writeCache = writeCache;
    this.readCache = readCache;
  }

  @Override
  public boolean isCacheEnabled() {
    return isCacheEnabled;
  }

  @Override
  public PythonReadCache getReadCache() {
    return readCache;
  }

  @Override
  public PythonWriteCache getWriteCache() {
    return writeCache;
  }

  public static CacheContextImpl of(SensorContext context, @Nullable SonarLintCache sonarLintCache) {
    String sonarModules = context.config().get("sonar.modules").orElse("");
    boolean isUsingSonarModules = !sonarModules.isEmpty();
    if (isUsingSonarModules && context.isCacheEnabled()) {
      LOG.info("Caching will be disabled for this analysis due to the use of the \"sonar.modules\" property.");
    }

    if (context.runtime().getApiVersion().isGreaterThanOrEqual(MINIMUM_RUNTIME_VERSION) &&
      !isUsingSonarModules) {
      if (context.runtime().getProduct().equals(SonarProduct.SONARLINT)) {
        if (sonarLintCache != null) {
          return new CacheContextImpl(true, new PythonWriteCacheImpl(sonarLintCache), new PythonReadCacheImpl(sonarLintCache));
        }
      } else {
        return new CacheContextImpl(context.isCacheEnabled(), new PythonWriteCacheImpl(context.nextCache()), new PythonReadCacheImpl(context.previousCache()));
      }
    }
    return dummyCache();
  }

  public static CacheContextImpl dummyCache() {
    return new CacheContextImpl(false, new DummyCache(), new DummyCache());
  }
}
