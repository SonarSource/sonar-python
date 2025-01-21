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
package org.sonar.plugins.python.caching;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import org.sonar.api.batch.sensor.cache.ReadCache;
import org.sonar.api.batch.sensor.cache.WriteCache;

public class TestWriteCache implements WriteCache {

  private final Map<String, byte[]> data = new HashMap<>();
  private ReadCache readCache;

  public TestWriteCache bind(ReadCache readCache) {
    this.readCache = readCache;
    return this;
  }

  public Map<String, byte[]> getData() {
    return data;
  }

  @Override
  public void write(String key, InputStream data) {
    try {
      write(key, data.readAllBytes());
    } catch (IOException e) {
      throw new IllegalStateException("Failed to read stream", e);
    }
  }

  @Override
  public void write(String key, byte[] data) {
    if (this.data.containsKey(key)) {
      throw new IllegalArgumentException(String.format("Same key cannot be written to multiple times (%s)", key));
    }
    this.data.put(key, data);
  }

  @Override
  public void copyFromPrevious(String key) {
    write(key, readCache.read(key));
  }
}
