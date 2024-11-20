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
package org.sonar.plugins.python.api.internal;

import org.sonar.api.Beta;
import org.sonar.plugins.python.api.caching.CacheContext;

/**
 * Common interface for providing callbacks that are triggered at the end of an analysis, after all files have been scanned.
 * <b>Warning: keeping state between files can lead to memory leaks. Implement with care.</b>
 */
@Beta
public interface EndOfAnalysis {

  /**
   * A method called when all files have been processed.
   * @param cacheContext CacheContext that can be used to store or retrieve information in the server cache.
   */
  void endOfAnalysis(CacheContext cacheContext);
}
