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

import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;

public class CacheContextImpl implements CacheContext {

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
}
