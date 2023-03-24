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

import com.google.common.annotations.Beta;
import java.io.File;
import java.util.Collection;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;

public interface SubscriptionContext {
  Tree syntaxNode();

  PythonCheck.PreciseIssue addIssue(Tree element, @Nullable String message);

  PythonCheck.PreciseIssue addIssue(LocationInFile location, @Nullable String message);

  PythonCheck.PreciseIssue addIssue(Token token, @Nullable String message);

  PythonCheck.PreciseIssue addIssue(Token from, Token to, @Nullable String message);

  PythonCheck.PreciseIssue addFileIssue(String finalMessage);

  PythonCheck.PreciseIssue addLineIssue(String message, int lineNumber);

  PythonFile pythonFile();

  /**
   * List of Python versions this project is compatible with.
   */
  @Beta
  Set<PythonVersionUtils.Version> currentPythonVersions();

  /**
   * Returns symbols declared in stub files (e.g. typeshed) used in the analyzed project.
   */
  @Beta
  Collection<Symbol> stubFilesSymbols();

  /**
   * Returns null in case of Sonarlint context
   */
  @CheckForNull
  File workingDirectory();

  @Beta
  CacheContext cacheContext();
}
