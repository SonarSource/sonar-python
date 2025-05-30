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
package org.sonar.plugins.python.api.caching;

import org.sonar.api.Beta;

@Beta
public interface PythonWriteCache {
  /**
   * Save a new entry in the cache.
   * @throws {@code IllegalArgumentException} if the cache already contains the key.
   */
  void write(String key, byte[] data);

  /**
   * Copy a cached entry from the previous cache to the new one.
   * @throws {@code IllegalArgumentException} if the previous cache doesn't contain given key or if this cache already contains the key.
   */
  void copyFromPrevious(String key);
}
