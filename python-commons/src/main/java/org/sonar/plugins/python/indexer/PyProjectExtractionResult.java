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

import java.io.File;
import java.util.List;

/**
 * Result of extracting source roots from pyproject.toml, including the build system detected.
 *
 * @param configRoots The extracted source roots
 * @param buildSystem The build system that provided the source roots
 */
public record PyProjectExtractionResult(ConfigSourceRoots configRoots, PackageResolutionResult.BuildSystem buildSystem){

  public static PyProjectExtractionResult empty(File file) {
    return new PyProjectExtractionResult(ConfigSourceRoots.empty(file), PackageResolutionResult.BuildSystem.NONE);
  }

  public boolean hasRoots() {
    return !configRoots.relativeRoots().isEmpty();
  }

  public List<String> relativeRoots(){
    return configRoots.relativeRoots();
  }
}
