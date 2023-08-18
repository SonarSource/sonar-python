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

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.api.batch.sensor.cache.WriteCache;

public class PythonWriteCacheImplTest {

  @Test
  public void write() {
    byte[] bytes = "hello".getBytes();
    WriteCache writeCache = Mockito.spy(WriteCache.class);

    PythonWriteCacheImpl pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    pythonWriteCache.write("key", bytes);

    Mockito.verify(writeCache, Mockito.times(1))
      .write("key", bytes);
  }

  @Test
  public void copy_from_previous() {
    WriteCache writeCache = Mockito.spy(WriteCache.class);

    PythonWriteCacheImpl pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    pythonWriteCache.copyFromPrevious("key");

    Mockito.verify(writeCache, Mockito.times(1))
      .copyFromPrevious("key");
  }
}
