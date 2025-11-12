/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.plugins.python.api;

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

import org.sonar.api.scanner.ScannerSide;
import org.sonarsource.api.sonarlint.SonarLintSide;

@SonarLintSide
@ScannerSide
public class SonarLintCacheWrapper {

  private final SonarLintCache sonarLintCache;

  public SonarLintCacheWrapper() {
    // Constructor to be used by pico if no SonarLintCache are to be found and injected.
    this(null);
  }

  public SonarLintCacheWrapper(@Nullable SonarLintCache sonarLintCache) {
    this.sonarLintCache = sonarLintCache;
  }

  @CheckForNull
  public SonarLintCache sonarLintCache() {
    return sonarLintCache;
  }
}
