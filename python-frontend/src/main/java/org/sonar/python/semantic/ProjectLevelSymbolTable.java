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
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.DescriptorUtils;
import org.sonar.python.index.VariableDescriptor;

import static org.sonar.python.tree.TreeUtils.getSymbolFromTree;
import static org.sonar.python.tree.TreeUtils.nthArgumentOrKeyword;

public class ProjectLevelSymbolTable {

  private final Map<String, Set<Descriptor>> globalDescriptorsByModuleName;
  private Map<String, Descriptor> globalDescriptorsByFQN;
  private final Set<String> djangoViewsFQN = new HashSet<>();
  private Set<String> queriedSymbolNames = new HashSet<>();

  public static ProjectLevelSymbolTable empty() {
    return new ProjectLevelSymbolTable(Collections.emptyMap());
  }

  public static ProjectLevelSymbolTable from(Map<String, Set<Symbol>> globalSymbolsByModuleName) {
    return new ProjectLevelSymbolTable(globalSymbolsByModuleName);
  }

  public ProjectLevelSymbolTable() {
    this.globalDescriptorsByModuleName = new HashMap<>();
  }

  private ProjectLevelSymbolTable(Map<String, Set<Symbol>> globalSymbolsByModuleName) {
    this.globalDescriptorsByModuleName = new HashMap<>();
    globalSymbolsByModuleName.entrySet().forEach(entry -> {
      String moduleName = entry.getKey();
      Set<Symbol> symbols = entry.getValue();
      Set<Descriptor> globalDescriptors = symbols.stream().map(DescriptorUtils::descriptor).collect(Collectors.toSet());
      globalDescriptorsByModuleName.put(moduleName, globalDescriptors);
    });
  }

  public void removeModule(String packageName, String fileName) {
    String fullyQualifiedModuleName = SymbolUtils.fullyQualifiedModuleName(packageName, fileName);
    globalDescriptorsByModuleName.remove(fullyQualifiedModuleName);
    // ensure globalDescriptorsByFQN is re-computed
    this.globalDescriptorsByFQN = null;
  }

  public void addModule(FileInput fileInput, String packageName, PythonFile pythonFile) {
    SymbolTableBuilder symbolTableBuilder = new SymbolTableBuilder(packageName, pythonFile);
    String fullyQualifiedModuleName = SymbolUtils.fullyQualifiedModuleName(packageName, pythonFile.fileName());
    fileInput.accept(symbolTableBuilder);
    Set<Descriptor> globalDescriptors = new HashSet<>();
    for (Symbol globalVariable : fileInput.globalVariables()) {
      String fullyQualifiedVariableName = globalVariable.fullyQualifiedName();
      if (((fullyQualifiedVariableName != null) && !fullyQualifiedVariableName.startsWith(fullyQualifiedModuleName)) ||
        globalVariable.usages().stream().anyMatch(u -> u.kind().equals(Usage.Kind.IMPORT))) {
        // TODO: We don't put builtin or imported names in global symbol table to avoid duplicate FQNs in project level symbol table (to fix with SONARPY-647)
        continue;
      }
      if (globalVariable.is(Symbol.Kind.CLASS, Symbol.Kind.FUNCTION)) {
        globalDescriptors.add(DescriptorUtils.descriptor(globalVariable));
      } else {
        String fullyQualifiedName = fullyQualifiedModuleName + "." + globalVariable.name();
        if (globalVariable.is(Symbol.Kind.AMBIGUOUS)) {
          globalDescriptors.add(DescriptorUtils.ambiguousDescriptor((AmbiguousSymbol) globalVariable, fullyQualifiedName));
        } else {
          globalDescriptors.add(new VariableDescriptor(globalVariable.name(), fullyQualifiedName, globalVariable.annotatedTypeName()));
        }
      }
    }
    globalDescriptorsByModuleName.put(fullyQualifiedModuleName, globalDescriptors);
    if (globalDescriptorsByFQN != null) {
      // TODO: build globalSymbolsByFQN incrementally
      addModuleToGlobalSymbolsByFQN(globalDescriptors);
    }
    DjangoViewsVisitor djangoViewsVisitor = new DjangoViewsVisitor();
    fileInput.accept(djangoViewsVisitor);
  }

  private void addModuleToGlobalSymbolsByFQN(Set<Descriptor> descriptors) {
    Map<String, Descriptor> moduleDescriptorsByFQN = descriptors.stream()
      .filter(d -> d.fullyQualifiedName() != null)
      .collect(Collectors.toMap(Descriptor::fullyQualifiedName, Function.identity(), AmbiguousDescriptor::create));
    globalDescriptorsByFQN.putAll(moduleDescriptorsByFQN);

  }

  private Map<String, Descriptor> globalDescriptorsByFQN() {
    if (globalDescriptorsByFQN == null) {
      globalDescriptorsByFQN = globalDescriptorsByModuleName.values()
        .stream()
        .flatMap(Collection::stream)
        .filter(descriptor -> descriptor.fullyQualifiedName() != null)
        .collect(Collectors.toMap(Descriptor::fullyQualifiedName, Function.identity(), AmbiguousDescriptor::create));
    }
    return globalDescriptorsByFQN;
  }

  @CheckForNull
  public Symbol getSymbol(@Nullable String fullyQualifiedName) {
    return getSymbol(fullyQualifiedName, null);
  }

  public Symbol getSymbol(@Nullable String fullyQualifiedName, @Nullable String localSymbolName) {
    if (fullyQualifiedName == null) return null;
    if (queriedSymbolNames.contains(fullyQualifiedName)) {
      // cyclic dependencies
      queriedSymbolNames = new HashSet<>();
      String[] fqnSplitByDot = fullyQualifiedName.split("\\.");
      localSymbolName =  localSymbolName != null ? localSymbolName : fqnSplitByDot[fqnSplitByDot.length - 1];
      return new SymbolImpl(localSymbolName, fullyQualifiedName);
    }
    Descriptor descriptor = globalDescriptorsByFQN().get(fullyQualifiedName);
    if (descriptor == null) {
      queriedSymbolNames = new HashSet<>();
      return null;
    } else {
      queriedSymbolNames.add(fullyQualifiedName);
      Symbol symbol = DescriptorUtils.symbolFromDescriptor(descriptor, this, localSymbolName, new HashMap<>());
      queriedSymbolNames = new HashSet<>();
      return symbol;
    }
  }

  @CheckForNull
  public Set<Symbol> getSymbolsFromModule(@Nullable String moduleName) {
    Set<Descriptor> descriptors = globalDescriptorsByModuleName.get(moduleName);
    Map<String, Symbol> createdSymbols = new HashMap<>();
    if (descriptors == null) {
      return null;
    }
    return descriptors.stream()
      .map(desc -> DescriptorUtils.symbolFromDescriptor(desc, this, null, createdSymbols)).collect(Collectors.toSet());
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
