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

import org.sonar.api.batch.sensor.cache.WriteCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;

public class PythonWriteCacheImpl implements PythonWriteCache {

  private WriteCache writeCache;

  public PythonWriteCacheImpl(WriteCache writeCache) {
    this.writeCache = writeCache;
  }

  @Override
  public void write(String key, byte[] data) {
    this.writeCache.write(key, data);
  }

  @Override
  public void copyFromPrevious(String key) {
    this.writeCache.copyFromPrevious(key);
  }
}
