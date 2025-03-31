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
package org.sonar.python.semantic;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.DescriptorUtils;
import org.sonar.python.semantic.v2.BasicTypeTable;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.semantic.v2.converter.PythonTypeToDescriptorConverter;
import org.sonar.python.semantic.v2.typeshed.TypeShedDescriptorsProvider;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeChecker;

import static org.sonar.python.tree.TreeUtils.nthArgumentOrKeyword;

public class ProjectLevelSymbolTable {

  private final PythonTypeToDescriptorConverter pythonTypeToDescriptorConverter;
  private final Map<String, Set<Descriptor>> globalDescriptorsByModuleName;
  private Map<String, Descriptor> globalDescriptorsByFQN;
  private final Set<String> djangoViewsFQN;
  private final Map<String, Set<String>> importsByModule;
  private final Set<String> projectBasePackages;
  private TypeShedDescriptorsProvider typeShedDescriptorsProvider = null;
  private Set<Symbol> cachedSymbols = null;

  public static ProjectLevelSymbolTable empty() {
    return new ProjectLevelSymbolTable();
  }

  public static ProjectLevelSymbolTable from(Map<String, Set<Descriptor>> globalDescriptorsByModuleName) {
    var projectLevelSymbolTable = ProjectLevelSymbolTable.empty();

    for (var entry : globalDescriptorsByModuleName.entrySet()) {
      var descriptors = entry.getValue();
      projectLevelSymbolTable.globalDescriptorsByModuleName.put(entry.getKey(), descriptors);
    }

    return projectLevelSymbolTable;
  }

  private ProjectLevelSymbolTable() {
    this.pythonTypeToDescriptorConverter = new PythonTypeToDescriptorConverter();
    this.globalDescriptorsByModuleName = new ConcurrentHashMap<>();
    this.djangoViewsFQN = new HashSet<>();
    this.importsByModule = new ConcurrentHashMap<>();
    this.projectBasePackages = new HashSet<>();
  }

  public void removeModule(String packageName, String fileName) {
    String fullyQualifiedModuleName = SymbolUtils.fullyQualifiedModuleName(packageName, fileName);
    globalDescriptorsByModuleName.remove(fullyQualifiedModuleName);
    // ensure globalDescriptorsByFQN is re-computed
    this.globalDescriptorsByFQN = null;
  }

  public void addModule(FileInput fileInput, String packageName, PythonFile pythonFile) {
    String fullyQualifiedModuleName = SymbolUtils.fullyQualifiedModuleName(packageName, pythonFile.fileName());
    var symbolTable = new SymbolTableBuilderV2(fileInput).build();
    var typeInferenceV2 = new TypeInferenceV2(new BasicTypeTable(new ProjectLevelTypeTable(this)), pythonFile, symbolTable, packageName);
    var typesBySymbol = typeInferenceV2.inferTopLevelTypes(fileInput);
    importsByModule.put(fullyQualifiedModuleName, typeInferenceV2.importedModulesFQN());
    var moduleDescriptors = typesBySymbol.entrySet()
      .stream()
      .filter(entry -> isNotMissingType(entry.getValue()))
      .map(entry -> {
          var descriptor = pythonTypeToDescriptorConverter.convert(fullyQualifiedModuleName, entry.getKey(), entry.getValue());
          return Map.entry(entry.getKey(), descriptor);
        }
      )
      .filter(entry -> !(!Objects.requireNonNull(entry.getValue().fullyQualifiedName()).startsWith(fullyQualifiedModuleName)
        || entry.getKey().usages().stream().anyMatch(u -> u.kind().equals(UsageV2.Kind.IMPORT))))
      .map(Map.Entry::getValue)
      .collect(Collectors.toSet());
    globalDescriptorsByModuleName.merge(fullyQualifiedModuleName, moduleDescriptors, ProjectLevelSymbolTable::mergeDescriptors);
    addModuleToGlobalSymbolsByFQN(moduleDescriptors);

    DjangoViewsVisitor djangoViewsVisitor = new DjangoViewsVisitor(fullyQualifiedModuleName);
    fileInput.accept(djangoViewsVisitor);
  }

  private static Set<Descriptor> mergeDescriptors(Set<Descriptor> oldDescriptors, Set<Descriptor> newDescriptors) {
    var oldDescriptorsByFqn = oldDescriptors.stream()
      .collect(Collectors.toMap(Descriptor::fullyQualifiedName, Function.identity(), AmbiguousDescriptor::create));
    var newDescriptorsByFqn = newDescriptors.stream()
      .collect(Collectors.toMap(Descriptor::fullyQualifiedName, Function.identity(), AmbiguousDescriptor::create));
    var mergedDescriptorsByFqn = mergeDescriptors(oldDescriptorsByFqn, newDescriptorsByFqn);
    return Set.copyOf(mergedDescriptorsByFqn.values());
  }

  private static Map<String, Descriptor> mergeDescriptors(Map<String, Descriptor> oldDescriptorsByFqn,
    Map<String, Descriptor> newDescriptorsByFqn) {
    var mergedMap = new HashMap<>(oldDescriptorsByFqn);
    newDescriptorsByFqn.forEach((fqn, descriptor) -> mergedMap.merge(fqn, descriptor, ProjectLevelSymbolTable::mergeDescriptors));
    return mergedMap;
  }

  @CheckForNull
  private static Descriptor mergeDescriptors(@Nullable Descriptor oldDescriptor, @Nullable Descriptor newDescriptor) {
    if (oldDescriptor == newDescriptor) {
      return oldDescriptor;
    }
    return AmbiguousDescriptor.create(oldDescriptor, newDescriptor);
  }

  private static boolean isNotMissingType(Set<PythonType> types) {
    return !types.isEmpty() && types.stream().noneMatch(UnknownType.UnresolvedImportType.class::isInstance);
  }

  private void addModuleToGlobalSymbolsByFQN(Set<Descriptor> descriptors) {
    var globalDescriptors = this.globalDescriptorsByFQN();
    var moduleDescriptorsByFQN = descriptors.stream()
      .filter(d -> d.fullyQualifiedName() != null)
      .collect(Collectors.toMap(Descriptor::fullyQualifiedName, Function.identity(), AmbiguousDescriptor::create));
    moduleDescriptorsByFQN.forEach((fqn, descriptor) -> globalDescriptors.merge(fqn, descriptor, ProjectLevelSymbolTable::mergeDescriptors));
  }

  private synchronized Map<String, Descriptor> globalDescriptorsByFQN() {
    if (globalDescriptorsByFQN == null) {
      globalDescriptorsByFQN = globalDescriptorsByModuleName.values()
        .stream()
        .flatMap(Collection::stream)
        .filter(descriptor -> descriptor.fullyQualifiedName() != null)
        .collect(Collectors.toConcurrentMap(Descriptor::fullyQualifiedName, Function.identity(), AmbiguousDescriptor::create));
    }
    return globalDescriptorsByFQN;
  }

  @CheckForNull
  public Symbol getSymbol(@Nullable String fullyQualifiedName) {
    return getSymbol(fullyQualifiedName, null);
  }

  public Symbol getSymbol(@Nullable String fullyQualifiedName, @Nullable String localSymbolName) {
    return getSymbol(fullyQualifiedName, localSymbolName, new HashMap<>(), new HashMap<>());
  }

  public Symbol getSymbol(@Nullable String fullyQualifiedName, @Nullable String localSymbolName,
                          Map<Descriptor, Symbol> createdSymbolsByDescriptor, Map<String, Symbol> createdSymbolsByFqn) {
    if (fullyQualifiedName == null) return null;
    Descriptor descriptor = globalDescriptorsByFQN().get(fullyQualifiedName);
    return descriptor == null ? null : DescriptorUtils.symbolFromDescriptor(descriptor, this, localSymbolName, createdSymbolsByDescriptor, createdSymbolsByFqn);
  }

  @CheckForNull
  public Set<Symbol> getSymbolsFromModule(@Nullable String moduleName) {
    Set<Descriptor> descriptors = getDescriptorsFromModule(moduleName);
    if (descriptors == null) {
      return null;
    }
    Map<Descriptor, Symbol> createdSymbolsByDescriptor = new HashMap<>();
    Map<String, Symbol> createdSymbolsByFqn = new HashMap<>();
    return descriptors.stream()
      .map(desc -> DescriptorUtils.symbolFromDescriptor(desc, this, null, createdSymbolsByDescriptor, createdSymbolsByFqn)).collect(Collectors.toSet());
  }

  @CheckForNull
  public Set<Descriptor> getDescriptorsFromModule(@Nullable String moduleName) {
    if (moduleName == null) {
      return null;
    }
    return globalDescriptorsByModuleName.get(moduleName);
  }

  public Map<String, Set<String>> importsByModule() {
    return Collections.unmodifiableMap(importsByModule);
  }

  public void insertEntry(String moduleName, Set<Descriptor> descriptors) {
    this.globalDescriptorsByModuleName.put(moduleName, descriptors);
  }

  @CheckForNull
  public Set<Descriptor> descriptorsForModule(String moduleName) {
    return globalDescriptorsByModuleName.get(moduleName);
  }

  private synchronized void addDjangoView(String fqn) {
    djangoViewsFQN.add(fqn);
  }

  public boolean isDjangoView(@Nullable String fqn) {
    return djangoViewsFQN.contains(fqn);
  }

  public synchronized void addProjectPackage(String projectPackage) {
    projectBasePackages.add(projectPackage.split("\\.", 2)[0]);
  }

  public Set<String> projectBasePackages() {
    return projectBasePackages;
  }

  public TypeShedDescriptorsProvider typeShedDescriptorsProvider() {
    if (typeShedDescriptorsProvider == null) {
      typeShedDescriptorsProvider = new TypeShedDescriptorsProvider(projectBasePackages);
    }
    return typeShedDescriptorsProvider;
  }

  /**
   * Returns stub symbols to be used by SonarSecurity.
   * Ambiguous symbols that only contain class symbols are disambiguated with latest Python version.
   */
  public Collection<Symbol> stubFilesSymbols() {
    if (cachedSymbols != null) {
      return cachedSymbols;
    }
    Map<String, Symbol> symbolsByFqn = new HashMap<>();
    cachedSymbols = new HashSet<>();
    for (Descriptor descriptor : typeShedDescriptorsProvider.stubFilesDescriptors()) {
      if (descriptor.fullyQualifiedName() != null) {
        Symbol symbol = symbolsByFqn.computeIfAbsent(descriptor.fullyQualifiedName(), k ->
          DescriptorUtils.symbolFromDescriptor(descriptor, this, null, new HashMap<>(), new HashMap<>()));
        cachedSymbols.add(symbol);
      }
    }
    return cachedSymbols;
  }

  private class DjangoViewsVisitor extends BaseTreeVisitor {

    String fullyQualifiedModuleName;
    private TypeCheckBuilder confPathCall = null;
    private TypeCheckBuilder pathCall = null;

    public DjangoViewsVisitor(String fullyQualifiedModuleName) {
      this.fullyQualifiedModuleName = fullyQualifiedModuleName;
    }

    @Override
    public void visitFileInput(FileInput fileInput) {
      TypeChecker typeChecker = new TypeChecker(new BasicTypeTable(new ProjectLevelTypeTable(ProjectLevelSymbolTable.this)));
      confPathCall = typeChecker.typeCheckBuilder().isTypeWithName("django.urls.conf.path");
      pathCall = typeChecker.typeCheckBuilder().isTypeWithName("django.urls.path");
      super.visitFileInput(fileInput);
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      super.visitCallExpression(callExpression);
      if (isCallRegisteringDjangoView(callExpression)) {
        RegularArgument viewArgument = nthArgumentOrKeyword(1, "view", callExpression.arguments());
        if (viewArgument != null) {
          PythonType pythonType = viewArgument.expression().typeV2();
          if (pythonType instanceof UnknownType.UnresolvedImportType unresolvedImportType) {
            String importPath = unresolvedImportType.importPath();
            addDjangoView(importPath);
          } else if (pythonType instanceof FunctionType functionType) {
            addDjangoView(functionType.fullyQualifiedName());
          }
        }
      }
    }

    private boolean isCallRegisteringDjangoView(CallExpression callExpression) {
      TriBool isConfPathCall = confPathCall.check(callExpression.callee().typeV2());
      TriBool isPathCall = pathCall.check(callExpression.callee().typeV2());
      return isConfPathCall.equals(TriBool.TRUE) || isPathCall.equals(TriBool.TRUE);
    }
  }
}
