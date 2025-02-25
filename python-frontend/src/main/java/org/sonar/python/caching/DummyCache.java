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
package org.sonar.python.caching;

import java.io.InputStream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;

public class DummyCache implements PythonReadCache, PythonWriteCache {

  @Override
  public InputStream read(String key) {
    throw new IllegalArgumentException("No cache data available");
  }

  @CheckForNull
  @Override
  public byte[] readBytes(String key) {
    return null;
  }

  @Override
  public boolean contains(String key) {
    return false;
  }

  @Override
  public void write(String key, byte[] data) {
    throw new IllegalArgumentException(String.format("Same key cannot be written to multiple times (%s)", key));
  }

  @Override
  public void copyFromPrevious(String key) {
    throw new IllegalArgumentException("No cache data available");
  }
}

