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

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.api.batch.sensor.cache.WriteCache;

class PythonWriteCacheImplTest {

  @Test
  void write() {
    byte[] bytes = "hello".getBytes();
    WriteCache writeCache = Mockito.spy(WriteCache.class);

    PythonWriteCacheImpl pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    pythonWriteCache.write("key", bytes);

    Mockito.verify(writeCache, Mockito.times(1))
      .write("key", bytes);
  }

  @Test
  void copy_from_previous() {
    WriteCache writeCache = Mockito.spy(WriteCache.class);

    PythonWriteCacheImpl pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    pythonWriteCache.copyFromPrevious("key");

    Mockito.verify(writeCache, Mockito.times(1))
      .copyFromPrevious("key");
  }
}
