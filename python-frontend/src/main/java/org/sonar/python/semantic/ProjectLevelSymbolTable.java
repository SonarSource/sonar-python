/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.semantic;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.RegularArgument;

import static org.sonar.python.tree.TreeUtils.getSymbolFromTree;
import static org.sonar.python.tree.TreeUtils.nthArgumentOrKeyword;

public class ProjectLevelSymbolTable {

  private final Map<String, Set<Symbol>> globalSymbolsByModuleName;
  private Map<String, Symbol> globalSymbolsByFQN;
  private final Set<String> djangoViewsFQN = new HashSet<>();

  public static ProjectLevelSymbolTable empty() {
    return new ProjectLevelSymbolTable(Collections.emptyMap());
  }

  public static ProjectLevelSymbolTable from(Map<String, Set<Symbol>> globalSymbolsByModuleName) {
    return new ProjectLevelSymbolTable(globalSymbolsByModuleName);
  }

  public ProjectLevelSymbolTable() {
    this.globalSymbolsByModuleName = new HashMap<>();
  }

  private ProjectLevelSymbolTable(Map<String, Set<Symbol>> globalSymbolsByModuleName) {
    this.globalSymbolsByModuleName = new HashMap<>(globalSymbolsByModuleName);
  }

  public void addModule(FileInput fileInput, String packageName, PythonFile pythonFile) {
    SymbolTableBuilder symbolTableBuilder = new SymbolTableBuilder(packageName, pythonFile);
    String fullyQualifiedModuleName = SymbolUtils.fullyQualifiedModuleName(packageName, pythonFile.fileName());
    fileInput.accept(symbolTableBuilder);
    Set<Symbol> globalSymbols = new HashSet<>();
    for (Symbol globalVariable : fileInput.globalVariables()) {
      String fullyQualifiedVariableName = globalVariable.fullyQualifiedName();
      if (((fullyQualifiedVariableName != null) && !fullyQualifiedVariableName.startsWith(fullyQualifiedModuleName)) ||
        globalVariable.usages().stream().anyMatch(u -> u.kind().equals(Usage.Kind.IMPORT))) {
        // TODO: We don't put builtin or imported names in global symbol table to avoid duplicate FQNs in project level symbol table (to fix with SONARPY-647)
        continue;
      }
      if (globalVariable.kind() == Symbol.Kind.CLASS) {
        globalSymbols.add(((ClassSymbolImpl) globalVariable).copyWithoutUsages());
      } else if (globalVariable.kind() == Symbol.Kind.FUNCTION) {
        globalSymbols.add(new FunctionSymbolImpl(globalVariable.name(), ((FunctionSymbol) globalVariable)));
      } else {
        globalSymbols.add(new SymbolImpl(globalVariable.name(), fullyQualifiedModuleName + "." + globalVariable.name(), globalVariable.annotatedTypeName()));
      }
    }
    globalSymbolsByModuleName.put(fullyQualifiedModuleName, globalSymbols);
    DjangoViewsVisitor djangoViewsVisitor = new DjangoViewsVisitor();
    fileInput.accept(djangoViewsVisitor);
  }

  private Map<String, Symbol> globalSymbolsByFQN() {
    if (globalSymbolsByFQN == null) {
      globalSymbolsByFQN = globalSymbolsByModuleName.values()
        .stream()
        .flatMap(Collection::stream)
        .filter(symbol -> symbol.fullyQualifiedName() != null)
        .collect(Collectors.toMap(Symbol::fullyQualifiedName, Function.identity(), AmbiguousSymbolImpl::create));
    }
    return globalSymbolsByFQN;
  }

  @CheckForNull
  public Symbol getSymbol(@Nullable String fullyQualifiedName) {
    return globalSymbolsByFQN().get(fullyQualifiedName);
  }

  @CheckForNull
  public Set<Symbol> getSymbolsFromModule(@Nullable String moduleName) {
    return globalSymbolsByModuleName.get(moduleName);
  }

  public boolean isDjangoView(@Nullable String fqn) {
    return djangoViewsFQN.contains(fqn);
  }

  private class DjangoViewsVisitor extends BaseTreeVisitor {
    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol == null) {
        return;
      }
      if ("django.urls.conf.path".equals(calleeSymbol.fullyQualifiedName())) {
        RegularArgument viewArgument = nthArgumentOrKeyword(1, "view", callExpression.arguments());
        if (viewArgument != null) {
          getSymbolFromTree(viewArgument.expression())
            .filter(symbol -> symbol.fullyQualifiedName() != null)
            .ifPresent(symbol -> djangoViewsFQN.add(symbol.fullyQualifiedName()));
        }
      }
    }
  }
}
