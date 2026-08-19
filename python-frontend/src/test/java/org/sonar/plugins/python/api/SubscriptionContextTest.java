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

import java.io.File;
import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;
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

import static org.assertj.core.api.Assertions.assertThat;

class SubscriptionContextTest {

  @Test
  void is_likely_test_file_defaults_to_false_for_external_implementations() {
    // A SubscriptionContext implemented outside the analyzer that predates isLikelyTestFile()
    // must keep compiling; the default implementation answers false.
    SubscriptionContext context = new MinimalSubscriptionContext();

    assertThat(context.isLikelyTestFile()).isFalse();
  }

  private static class MinimalSubscriptionContext implements SubscriptionContext {
    @Override
    public Tree syntaxNode() {
      return null;
    }

    @Override
    public PythonCheck.PreciseIssue addIssue(Tree element, String message) {
      return null;
    }

    @Override
    public PythonCheck.PreciseIssue addIssue(LocationInFile location, String message) {
      return null;
    }

    @Override
    public PythonCheck.PreciseIssue addIssue(Token token, String message) {
      return null;
    }

    @Override
    public PythonCheck.PreciseIssue addIssue(Token from, Token to, String message) {
      return null;
    }

    @Override
    public PythonCheck.PreciseIssue addFileIssue(String finalMessage) {
      return null;
    }

    @Override
    public PythonCheck.PreciseIssue addLineIssue(String message, int lineNumber) {
      return null;
    }

    @Override
    public PythonFile pythonFile() {
      return null;
    }

    @Override
    public Set<PythonVersionUtils.Version> sourcePythonVersions() {
      return null;
    }

    @Override
    public Collection<Symbol> stubFilesSymbols() {
      return null;
    }

    @Override
    public File workingDirectory() {
      return null;
    }

    @Override
    public CacheContext cacheContext() {
      return null;
    }

    @Override
    public TypeChecker typeChecker() {
      return null;
    }

    @Override
    public TypeTable typeTable() {
      return null;
    }

    @Override
    public ProjectConfiguration projectConfiguration() {
      return null;
    }

    @Override
    public CallGraph callGraph() {
      return null;
    }

    @Override
    public ControlFlowGraph cfg(Tree tree) {
      return null;
    }

    @Override
    public LiveVariablesAnalysis lva(Tree tree) {
      return null;
    }

    @Override
    public Set<Expression> valuesAtLocation(Name name) {
      return null;
    }

    @Override
    public Optional<DjangoViewInfo> getDjangoViewInfo(String fqn) {
      return Optional.empty();
    }
  }
}
