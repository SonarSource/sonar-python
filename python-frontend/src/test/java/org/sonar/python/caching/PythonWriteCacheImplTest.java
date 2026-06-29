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
package org.sonar.python.caching;

import org.junit.jupiter.api.Test;
import org.sonar.api.batch.sensor.cache.WriteCache;

import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class PythonWriteCacheImplTest {

  @Test
  void write() {
    byte[] bytes = "hello".getBytes();
    WriteCache writeCache = spy(WriteCache.class);

    PythonWriteCacheImpl pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    pythonWriteCache.write("key", bytes);

    verify(writeCache, times(1))
      .write("key", bytes);
  }

  @Test
  void copy_from_previous() {
    WriteCache writeCache = spy(WriteCache.class);

    PythonWriteCacheImpl pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    pythonWriteCache.copyFromPrevious("key");

    verify(writeCache, times(1))
      .copyFromPrevious("key");
  }
}
