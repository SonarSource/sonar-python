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
package org.sonar.python.semantic.v2;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.IntStream;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;

public class ProjectLevelTypeTable implements TypeTable {

  private final SymbolsModuleTypeProvider symbolsModuleTypeProvider;
  private final PythonType rootModule;
  private final LazyTypesContext lazyTypesContext;

  private final Lock createOrResolveSubModuleLock;

  public ProjectLevelTypeTable(ProjectLevelSymbolTable projectLevelSymbolTable) {
    this.lazyTypesContext = new LazyTypesContext(this);
    this.symbolsModuleTypeProvider = new SymbolsModuleTypeProvider(projectLevelSymbolTable, lazyTypesContext);
    this.rootModule = this.symbolsModuleTypeProvider.getRootModule();
    this.createOrResolveSubModuleLock = new ReentrantLock();
  }

  @Override
  public PythonType getBuiltinsModule() {
    return rootModule;
  }

  @Override
  public PythonType getType(String typeFqn) {
    return getType(typeFqn.split("\\."));
  }

  @Override
  public PythonType getType(String... typeFqnParts) {
    return getType(List.of(typeFqnParts));
  }

  @Override
  public PythonType getType(List<String> typeFqnParts) {
    var parent = rootModule;
    for (int i = 0; i < typeFqnParts.size(); i++) {
      var part = typeFqnParts.get(i);
      var moduleFqnParts = IntStream.rangeClosed(0, i)
        .mapToObj(typeFqnParts::get)
        .toList();
      if (parent instanceof ObjectType) {
        return PythonType.UNKNOWN;
      }
      Optional<PythonType> resolvedMember;
      if (parent instanceof ModuleType moduleType) {
        TypeWrapper typeWrapper = moduleType.members().get(part);
        if (typeWrapper instanceof LazyTypeWrapper lazyTypeWrapper && !lazyTypeWrapper.isResolved()) {
          if (shouldResolveImmediately(lazyTypeWrapper, typeFqnParts, i)) {
            // We try to resolve the type of the member if it points to a different module.
            // If it points to the same module, we try to resolve the submodule of the same name
            return typeWrapper.type();
          }

          // The member of the module is a LazyType, which means it's a re-exported type from a submodule
          // We try to resolve the submodule instead
          parent = createOrResolveSubModule(moduleFqnParts, moduleType);
          continue;
        }
      }
      resolvedMember = parent.resolveMember(part);
      if (resolvedMember.isPresent()) {
        parent = resolvedMember.get();
      } else if (parent instanceof ModuleType module) {
        parent = createOrResolveSubModule(moduleFqnParts, module);
      } else {
        return PythonType.UNKNOWN;
      }
    }
    return parent;
  }

  private static boolean shouldResolveImmediately(LazyTypeWrapper lazyTypeWrapper, List<String> typeFqnParts, int i) {
    return i == typeFqnParts.size() - 1 && !(lazyTypeWrapper.hasImportPath(String.join(".", typeFqnParts)));
  }

  /**
   * This method returns a module type for a given FQN, or unknown if it cannot be resolved.
   * It is to be used to retrieve modules referenced in the "from" clause of an "import from" statement,
   * as it will only consider submodules over package members in case of name conflict.
   */
  @Override
  public PythonType getModuleType(List<String> typeFqnParts) {
    var parent = rootModule;
    for (int i = 0; i < typeFqnParts.size(); i++) {
      var moduleFqnParts = IntStream.rangeClosed(0, i)
        .mapToObj(typeFqnParts::get)
        .toList();
      if (!(parent instanceof ModuleType moduleType)) {
        return PythonType.UNKNOWN;
      }
      parent = createOrResolveSubModule(moduleFqnParts, moduleType);
    }
    return parent;
  }

  private PythonType createOrResolveSubModule(List<String> moduleFqn, ModuleType parent) {
    try {
      createOrResolveSubModuleLock.lock();
      String part = moduleFqn.get(moduleFqn.size() - 1);
      return parent.resolveSubmodule(part).orElseGet(() -> symbolsModuleTypeProvider.convertModuleType(moduleFqn, parent));
    } finally {
      createOrResolveSubModuleLock.unlock();
    }
  }

  public LazyTypesContext lazyTypesContext() {
    return lazyTypesContext;
  }
}
