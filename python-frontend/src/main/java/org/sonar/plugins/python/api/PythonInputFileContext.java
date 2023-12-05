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
package org.sonar.plugins.python.api;

import java.io.File;
import java.util.Collection;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.api.Beta;
import org.sonar.api.SonarProduct;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.types.TypeShed;

public class PythonInputFileContext {

  private final PythonFile pythonFile;
  private final File workingDirectory;
  private final CacheContext cacheContext;

  private final SonarProduct sonarProduct;

  public PythonInputFileContext(PythonFile pythonFile, @Nullable File workingDirectory, CacheContext cacheContext, SonarProduct sonarProduct) {
    this.pythonFile = pythonFile;
    this.workingDirectory = workingDirectory;
    this.cacheContext = cacheContext;
    this.sonarProduct = sonarProduct;
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
    return TypeShed.stubFilesSymbols();
  }

  @CheckForNull
  public File workingDirectory() {
    return workingDirectory;
  }

  public SonarProduct sonarProduct() {
    return sonarProduct;
  }
}
