/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.LiveVariablesAnalysis;
import org.sonar.python.semantic.v2.callgraph.CallGraph;
import org.sonar.python.semantic.v2.typetable.TypeTable;
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
   * Reports whether the current file is likely to contain test code.
   * @return whether the file is likely test content
   */
  default boolean isLikelyTestFile() {
    return false;
  }

  /**
   * Returns normalized Python source-version compatibility buckets derived from the project configuration.
   * Configured versions older than 3.8 are represented by {@link PythonVersionUtils.Version#V_38}.
   * Version-sensitive rules should use these versions to determine which language features and policies apply.
   * They do not identify the serialized semantic models used during analysis; older source versions can be mapped
   * to a newer semantic model by {@link PythonVersionUtils#toSupportedSemanticVersions(Set)}.
   *
   * @return the normalized source-version compatibility buckets
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

  TypeTable typeTable();

  ProjectConfiguration projectConfiguration();

  CallGraph callGraph();

  ControlFlowGraph cfg(Tree tree);

  LiveVariablesAnalysis lva(Tree tree);

  Set<Expression> valuesAtLocation(Name name);

  /**
   * Returns Django view information for the given fully qualified function name.
   * @param fqn the fully qualified name of a function
   * @return Optional containing DjangoViewInfo if the function is a Django view, empty otherwise
   */
  Optional<DjangoViewInfo> getDjangoViewInfo(String fqn);
}
