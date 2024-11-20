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

import java.io.IOException;
import java.io.InputStream;
import javax.annotation.CheckForNull;
import org.sonar.api.batch.sensor.cache.ReadCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.caching.PythonReadCache;

public class PythonReadCacheImpl implements PythonReadCache {
  private static final Logger LOG = LoggerFactory.getLogger(PythonReadCacheImpl.class);

  private final ReadCache readCache;

  public PythonReadCacheImpl(ReadCache readCache) {
    this.readCache = readCache;
  }

  @Override
  public InputStream read(String key) {
    return readCache.read(key);
  }

  @CheckForNull
  @Override
  public byte[] readBytes(String key) {
    if (readCache.contains(key)) {
      try (var in = read(key)) {
        return in.readAllBytes();
      } catch (IOException e) {
        LOG.debug("Unable to read data for key: \"{}\"", key);
      }
    } else {
      LOG.trace("Cache miss for key '{}'", key);
    }
    return null;
  }

  @Override
  public boolean contains(String key) {
    return readCache.contains(key);
  }
}
