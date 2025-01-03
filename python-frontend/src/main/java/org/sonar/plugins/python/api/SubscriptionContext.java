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
import org.sonar.python.types.v2.TypeChecker;

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
  Set<PythonVersionUtils.Version> sourcePythonVersions();

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

  TypeChecker typeChecker();
}
