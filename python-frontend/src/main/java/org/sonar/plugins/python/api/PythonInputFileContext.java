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
package org.sonar.plugins.python.api;

import java.io.File;
import java.util.Collection;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.api.Beta;
import org.sonar.api.SonarProduct;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

public class PythonInputFileContext {

  private final PythonFile pythonFile;
  private final File workingDirectory;
  private final CacheContext cacheContext;

  private final SonarProduct sonarProduct;
  private final ProjectLevelSymbolTable projectLevelSymbolTable;

  public PythonInputFileContext(PythonFile pythonFile, @Nullable File workingDirectory, CacheContext cacheContext,
    SonarProduct sonarProduct, ProjectLevelSymbolTable projectLevelSymbolTable) {
    this.pythonFile = pythonFile;
    this.workingDirectory = workingDirectory;
    this.cacheContext = cacheContext;
    this.sonarProduct = sonarProduct;
    this.projectLevelSymbolTable = projectLevelSymbolTable;
  }

  public PythonInputFileContext(PythonFile pythonFile, @Nullable File workingDirectory, CacheContext cacheContext, ProjectLevelSymbolTable projectLevelSymbolTable) {
    this.pythonFile = pythonFile;
    this.workingDirectory = workingDirectory;
    this.cacheContext = cacheContext;
    this.sonarProduct = SonarProduct.SONARQUBE;
    this.projectLevelSymbolTable = projectLevelSymbolTable;
  }

  public PythonFile pythonFile() {
    return pythonFile;
  }

  @Beta
  public CacheContext cacheContext() {
    return cacheContext;
  }

  @Beta
  public Collection<Symbol> stubFilesSymbols() {
    return projectLevelSymbolTable.stubFilesSymbols();
  }

  @CheckForNull
  public File workingDirectory() {
    return workingDirectory;
  }

  public SonarProduct sonarProduct() {
    return sonarProduct;
  }
}
