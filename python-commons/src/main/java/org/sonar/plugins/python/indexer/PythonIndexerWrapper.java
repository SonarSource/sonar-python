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
package org.sonar.plugins.python.indexer;

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

import org.sonar.api.scanner.ScannerSide;
import org.sonarsource.api.sonarlint.SonarLintSide;

@ScannerSide
@SonarLintSide
public class PythonIndexerWrapper {

  private final PythonIndexer indexer;

  public PythonIndexerWrapper() {
    // Constructor to be used by pico if no indexer is to be found and injected.
    this(null);
  }

  public PythonIndexerWrapper(@Nullable PythonIndexer indexer) {
    this.indexer = indexer;
  }

  @CheckForNull
  public PythonIndexer indexer() {
    return indexer;
  }
}
