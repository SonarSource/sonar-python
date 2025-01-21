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

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import org.sonar.api.batch.sensor.cache.ReadCache;

public class TestReadCache implements ReadCache {
  private final Map<String, byte[]> data = new HashMap<>();

  @Override
  public InputStream read(String key) {
    if (!data.containsKey(key)) {
      throw new IllegalArgumentException(String.format("cache does not contain key \"%s\"", key));
    }
    byte[] buf = data.get(key);
    if (buf == null) {
      return new ByteArrayInputStream(new byte[0]);
    }
    return new ByteArrayInputStream(buf);
  }

  @Override
  public boolean contains(String key) {
    return data.containsKey(key);
  }

  public TestReadCache put(String key, byte[] data) {
    this.data.put(key, data);
    return this;
  }

  public TestReadCache putAll(Map<String, byte[]> data) {
    this.data.putAll(data);
    return this;
  }

  public TestReadCache putAll(TestWriteCache writeCache) {
    return this.putAll(writeCache.getData());
  }
}
