/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.types;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.BuiltinSymbols;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.protobuf.SymbolsProtos;
import org.sonar.python.types.protobuf.SymbolsProtos.ModuleSymbol;
import org.sonar.python.types.protobuf.SymbolsProtos.OverloadedFunctionSymbol;

import static org.sonar.plugins.python.api.types.BuiltinTypes.BOOL;
import static org.sonar.plugins.python.api.types.BuiltinTypes.COMPLEX;
import static org.sonar.plugins.python.api.types.BuiltinTypes.DICT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.FLOAT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.INT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.LIST;
import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;
import static org.sonar.plugins.python.api.types.BuiltinTypes.STR;
import static org.sonar.plugins.python.api.types.BuiltinTypes.TUPLE;

public class TypeShed {

  private static Map<String, Symbol> builtins;
  private static final Map<String, Map<String, Symbol>> typeShedSymbols = new HashMap<>();
  private static final Map<String, Set<Symbol>> builtinGlobalSymbols = new HashMap<>();
  private static final Set<String> modulesInProgress = new HashSet<>();

  private static final String PROTOBUF_CUSTOM_STUBS = "custom_protobuf/";
  private static final String PROTOBUF = "stdlib_protobuf/";
  private static final String PROTOBUF_THIRD_PARTY = "third_party_protobuf/";
  private static final String PROTOBUF_THIRD_PARTY_MYPY = "third_party_protobuf_mypy/";
  private static final String BUILTINS_FQN = "builtins";
  private static final String BUILTINS_PREFIX = BUILTINS_FQN + ".";
  // Those fundamentals builtins symbols need not to be ambiguous for the frontend to work properly
  private static final Set<String> BUILTINS_TO_DISAMBIGUATE = new HashSet<>(
    Arrays.asList(INT, FLOAT, COMPLEX, STR, BuiltinTypes.SET, DICT, LIST, TUPLE, NONE_TYPE, BOOL, "type", "super", "frozenset", "memoryview"));

  // This is needed for some Python 2 modules whose name differ from their Python 3 counterpart by capitalization only.
  private static final Map<String, String> MODULES_TO_DISAMBIGUATE = new HashMap<>();
  static {
    MODULES_TO_DISAMBIGUATE.put("ConfigParser", "2@ConfigParser");
    MODULES_TO_DISAMBIGUATE.put("Queue", "2@Queue");
    MODULES_TO_DISAMBIGUATE.put("SocketServer", "2@SocketServer");
  }

  static {
    BUILTINS_TO_DISAMBIGUATE.addAll(BuiltinSymbols.EXCEPTIONS);
  }

  private static final Logger LOG = LoggerFactory.getLogger(TypeShed.class);
  private static Set<String> supportedPythonVersions;
  private static ProjectLevelSymbolTable projectLevelSymbolTable;

  private TypeShed() {
  }

  //================================================================================
  // Public methods
  //================================================================================

  public static void setProjectLevelSymbolTable(ProjectLevelSymbolTable projectLevelSymbolTable) {
    TypeShed.projectLevelSymbolTable = projectLevelSymbolTable;
  }

  public static Map<String, Symbol> builtinSymbols() {
    if ((TypeShed.builtins == null)) {
      supportedPythonVersions = ProjectPythonVersion.currentVersions().stream().map(PythonVersionUtils.Version::serializedValue).collect(Collectors.toSet());
      Map<String, Symbol> builtins = getSymbolsFromProtobufModule(BUILTINS_FQN, PROTOBUF);
      builtins.put(NONE_TYPE, new ClassSymbolImpl(NONE_TYPE, NONE_TYPE));
      TypeShed.builtins = Collections.unmodifiableMap(builtins);
      TypeShed.builtinGlobalSymbols.put("", new HashSet<>(builtins.values()));
    }
    return builtins;
  }

  public static Map<String, Map<String, Symbol>> getLoadedTypeShedSymbols() {
    return typeShedSymbols;
  }

  public static ClassSymbol typeShedClass(String fullyQualifiedName) {
    Symbol symbol = builtinSymbols().get(fullyQualifiedName);
    if (symbol == null) {
      throw new IllegalArgumentException("No TypeShed symbol found for name: " + fullyQualifiedName);
    }
    if (symbol.kind() != Symbol.Kind.CLASS) {
      throw new IllegalArgumentException("TypeShed symbol " + fullyQualifiedName + " is not a class");
    }
    return (ClassSymbol) symbol;
  }

  private static boolean searchedModuleMatchesCurrentProject(String searchedModule) {
    if (projectLevelSymbolTable == null) {
      return false;
    }
    return projectLevelSymbolTable.projectBasePackages().contains(searchedModule.split("\\.", 2)[0]);
  }

  /**
   * Returns map of exported symbols by name for a given module
   */
  public static Map<String, Symbol> symbolsForModule(String moduleName) {
    if (searchedModuleMatchesCurrentProject(moduleName)) {
      return Collections.emptyMap();
    }
    if (!TypeShed.typeShedSymbols.containsKey(moduleName)) {
      Map<String, Symbol> symbols = searchTypeShedForModule(moduleName);
      typeShedSymbols.put(moduleName, symbols);
      return symbols;
    }
    return TypeShed.typeShedSymbols.get(moduleName);
  }

  @CheckForNull
  public static Symbol symbolWithFQN(String stdLibModuleName, String fullyQualifiedName) {
    Map<String, Symbol> symbols = symbolsForModule(stdLibModuleName);
    // TODO: improve performance - see SONARPY-955
    Symbol symbolByFqn = symbols.values().stream().filter(s -> fullyQualifiedName.equals(s.fullyQualifiedName())).findFirst().orElse(null);
    if (symbolByFqn != null || !fullyQualifiedName.contains(".")) {
      return symbolByFqn;
    }

    // If FQN of the member does not match the pattern of "package_name.file_name.symbol_name"
    // (e.g. it could be declared in package_name.file_name using import) or in case when
    // we have import with an alias (from module import method as alias_method), we retrieve symbol_name out of
    // FQN and try to look up by local symbol name, rather than FQN
    String[] fqnSplittedByDot = fullyQualifiedName.split("\\.");
    String symbolLocalNameFromFqn = fqnSplittedByDot[fqnSplittedByDot.length - 1];
    return symbols.get(symbolLocalNameFromFqn);
  }

  @CheckForNull
  public static Symbol symbolWithFQN(String fullyQualifiedName) {
    Map<String, Symbol> builtinSymbols = builtinSymbols();
    Symbol builtinSymbol = builtinSymbols.get(normalizedFqn(fullyQualifiedName));
    if (builtinSymbol != null) {
      return builtinSymbol;
    }
    String[] fqnSplittedByDot = fullyQualifiedName.split("\\.");
    String moduleName = Arrays.stream(fqnSplittedByDot, 0, fqnSplittedByDot.length - 1).collect(Collectors.joining("."));
    return symbolWithFQN(moduleName, fullyQualifiedName);
  }

  /**
   * Returns stub symbols to be used by SonarSecurity.
   * Ambiguous symbols that only contain class symbols are disambiguated with latest Python version.
   */
  public static Collection<Symbol> stubFilesSymbols() {
    Set<Symbol> symbols = new HashSet<>(TypeShed.builtinSymbols().values());
    for (Map<String, Symbol> symbolsByFqn : typeShedSymbols.values()) {
      for (Symbol symbol : symbolsByFqn.values()) {
        Symbol stubSymbol = symbol;
        if (isAmbiguousSymbolOfClasses(symbol)) {
          Symbol disambiguatedSymbol = disambiguateWithLatestPythonSymbol(((AmbiguousSymbol) symbol).alternatives());
          if (disambiguatedSymbol != null) {
            stubSymbol = disambiguatedSymbol;
          }
        }
        symbols.add(stubSymbol);
      }
    }
    return symbols;
  }

  public static Set<String> stubModules() {
    Set<String> modules = new HashSet<>();
    for (Map.Entry<String, Map<String, Symbol>> entry : typeShedSymbols.entrySet()) {
      if (!entry.getValue().isEmpty()) {
        modules.add(entry.getKey());
      }
    }
    return modules;
  }

  public static String normalizedFqn(String fqn) {
    if (fqn.startsWith(BUILTINS_PREFIX)) {
      return fqn.substring(BUILTINS_PREFIX.length());
    }
    return fqn;
  }

  public static String normalizedFqn(String fqn, String moduleName, String localName) {
    return normalizedFqn(fqn, moduleName, localName, null);
  }

  public static String normalizedFqn(String fqn, String moduleName, String localName, @Nullable String containerClassFqn) {
    if (containerClassFqn != null) return containerClassFqn + "." + localName;
    if (fqn.startsWith(moduleName)) return normalizedFqn(fqn);
    return moduleName + "." + localName;
  }

  public static boolean isValidForProjectPythonVersion(List<String> validForPythonVersions) {
    if (validForPythonVersions.isEmpty()) {
      return true;
    }
    // TODO: SONARPY-1522 - remove this workaround when we will have all the stubs for Python 3.12.
    Set<String> notSerializedVersions = PythonVersionUtils.getNotSerializedVersions().stream().map(PythonVersionUtils.Version::serializedValue).collect(Collectors.toSet());
    if (notSerializedVersions.containsAll(supportedPythonVersions)
        && validForPythonVersions.contains(PythonVersionUtils.Version.V_311.serializedValue())) {
      return true;
    }
    HashSet<String> intersection = new HashSet<>(validForPythonVersions);
    intersection.retainAll(supportedPythonVersions);
    return !intersection.isEmpty();
  }

  public static Set<Symbol> symbolsFromProtobufDescriptors(Set<Object> protobufDescriptors, @Nullable String containerClassFqn, String moduleName, boolean isFromClass) {
    Set<Symbol> symbols = new HashSet<>();
    for (Object descriptor : protobufDescriptors) {
      if (descriptor instanceof SymbolsProtos.ClassSymbol classSymbolProto) {
        symbols.add(new ClassSymbolImpl(classSymbolProto, moduleName));
      }
      if (descriptor instanceof SymbolsProtos.FunctionSymbol functionSymbolProto) {
        symbols.add(new FunctionSymbolImpl(functionSymbolProto, containerClassFqn, moduleName));
      }
      if (descriptor instanceof OverloadedFunctionSymbol overloadedFunctionSymbol) {
        if (overloadedFunctionSymbol.getDefinitionsList().size() < 2) {
          throw new IllegalStateException("Overloaded function symbols should have at least two definitions.");
        }
        symbols.add(fromOverloadedFunction(((OverloadedFunctionSymbol) descriptor), containerClassFqn, moduleName));
      }
      if (descriptor instanceof SymbolsProtos.VarSymbol varSymbol) {
        SymbolImpl symbol = new SymbolImpl(varSymbol, moduleName, isFromClass);
        if (varSymbol.getIsImportedModule()) {
          Map<String, Symbol> moduleExportedSymbols = symbolsForModule(varSymbol.getFullyQualifiedName());
          moduleExportedSymbols.values().forEach(symbol::addChildSymbol);
        }
        symbols.add(symbol);
      }
    }
    return symbols;
  }


  @CheckForNull
  public static SymbolsProtos.ClassSymbol classDescriptorWithFQN(String fullyQualifiedName) {
    String[] fqnSplitByDot = fullyQualifiedName.split("\\.");
    String symbolLocalNameFromFqn = fqnSplitByDot[fqnSplitByDot.length - 1];
    String moduleName = Arrays.stream(fqnSplitByDot, 0, fqnSplitByDot.length - 1).collect(Collectors.joining("."));
    InputStream resource = TypeShed.class.getResourceAsStream(PROTOBUF + moduleName + ".protobuf");
    if (resource == null) return null;
    ModuleSymbol moduleSymbol = deserializedModule(moduleName, resource);
    if (moduleSymbol == null) return null;
    for (SymbolsProtos.ClassSymbol classSymbol : moduleSymbol.getClassesList()) {
      if (classSymbol.getName().equals(symbolLocalNameFromFqn)) {
        return classSymbol;
      }
    }
    return null;
  }

  //================================================================================
  // Private methods
  //================================================================================

  // used by tests whenever 'sonar.python.version' changes
  public static void resetBuiltinSymbols() {
    builtins = null;
    typeShedSymbols.clear();
    builtinSymbols();
  }

  private static Map<String, Symbol> searchTypeShedForModule(String moduleName) {
    if (modulesInProgress.contains(moduleName)) {
      return new HashMap<>();
    }
    modulesInProgress.add(moduleName);
    Map<String, Symbol> customSymbols = getSymbolsFromProtobufModule(moduleName, PROTOBUF_CUSTOM_STUBS);
    if (!customSymbols.isEmpty()) {
      modulesInProgress.remove(moduleName);
      return customSymbols;
    }
    Map<String, Symbol> symbolsFromProtobuf = getSymbolsFromProtobufModule(moduleName, PROTOBUF);
    if (!symbolsFromProtobuf.isEmpty()) {
      modulesInProgress.remove(moduleName);
      return symbolsFromProtobuf;
    }

    Map<String, Symbol> thirdPartySymbolsMypy = getSymbolsFromProtobufModule(moduleName, PROTOBUF_THIRD_PARTY_MYPY);
    if (!thirdPartySymbolsMypy.isEmpty()) {
      modulesInProgress.remove(moduleName);
      return thirdPartySymbolsMypy;
    }

    Map<String, Symbol> thirdPartySymbols = getSymbolsFromProtobufModule(moduleName, PROTOBUF_THIRD_PARTY);
    modulesInProgress.remove(moduleName);
    return thirdPartySymbols;
  }

  /**
   * Some special symbols need NOT to be ambiguous for the frontend to work properly.
   * This method sort ambiguous symbol by python version and returns the one which is valid for
   * the most recent Python version.
   */
  @CheckForNull
  static Symbol disambiguateWithLatestPythonSymbol(Set<Symbol> alternatives) {
    int max = Integer.MIN_VALUE;
    Symbol latestPythonSymbol = null;
    for (Symbol alternative : alternatives) {
      int maxPythonVersionForSymbol = ((SymbolImpl) alternative).validForPythonVersions().stream().mapToInt(Integer::parseInt).max().orElse(max);
      if (maxPythonVersionForSymbol > max) {
        max = maxPythonVersionForSymbol;
        latestPythonSymbol = alternative;
      }
    }
    return latestPythonSymbol;
  }

  private static boolean isAmbiguousSymbolOfClasses(Symbol symbol) {
    if (symbol.is(Symbol.Kind.AMBIGUOUS)) {
      return ((AmbiguousSymbol) symbol).alternatives().stream().allMatch(s -> s.is(Symbol.Kind.CLASS));
    }
    return false;
  }

  private static Map<String, Symbol> getSymbolsFromProtobufModule(String moduleName, String dirName) {
    String fileName = MODULES_TO_DISAMBIGUATE.getOrDefault(moduleName, moduleName);
    InputStream resource = TypeShed.class.getResourceAsStream(dirName + fileName + ".protobuf");
    if (resource == null) {
      return Collections.emptyMap();
    }
    return getSymbolsFromProtobufModule(deserializedModule(moduleName, resource));
  }

  @CheckForNull
  static ModuleSymbol deserializedModule(String moduleName, InputStream resource) {
    try {
      return ModuleSymbol.parseFrom(resource);
    } catch (IOException e) {
      LOG.debug("Error while deserializing protobuf for module " + moduleName, e);
      return null;
    }
  }

  static Map<String, Symbol> getSymbolsFromProtobufModule(@Nullable ModuleSymbol moduleSymbol) {
    if (moduleSymbol == null) {
      return Collections.emptyMap();
    }

    // TODO: Use a common proxy interface Descriptor instead of using Object
    Map<String, Set<Object>> descriptorsByName = new HashMap<>();
    moduleSymbol.getClassesList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByName.computeIfAbsent(proto.getName(), d -> new HashSet<>()).add(proto));
    moduleSymbol.getFunctionsList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByName.computeIfAbsent(proto.getName(), d -> new HashSet<>()).add(proto));
    moduleSymbol.getOverloadedFunctionsList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByName.computeIfAbsent(proto.getName(), d -> new HashSet<>()).add(proto));
    moduleSymbol.getVarsList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByName.computeIfAbsent(proto.getName(), d -> new HashSet<>()).add(proto));

    Map<String, Symbol> deserializedSymbols = new HashMap<>();

    for (Map.Entry<String, Set<Object>> entry : descriptorsByName.entrySet()) {
      String name = entry.getKey();
      Set<Symbol> symbols = symbolsFromProtobufDescriptors(entry.getValue(), null, moduleSymbol.getFullyQualifiedName(), false);
      Symbol disambiguatedSymbol = disambiguateSymbolsWithSameName(name, symbols, moduleSymbol.getFullyQualifiedName());
      deserializedSymbols.put(name, disambiguatedSymbol);
    }
    return deserializedSymbols;
  }

  private static Symbol disambiguateSymbolsWithSameName(String name, Set<Symbol> symbols, String moduleFqn) {
    if (symbols.size() > 1) {
      if (haveAllTheSameFqn(symbols) && !isBuiltinToDisambiguate(moduleFqn, name)) {
        return AmbiguousSymbolImpl.create(symbols);
      }
      if (!moduleFqn.equals(BUILTINS_FQN)) {
        String fqns = symbols.stream()
          .map(Symbol::fullyQualifiedName)
          .map(fqn -> fqn == null ? "N/A" : fqn)
          .collect(Collectors.joining(","));
        LOG.debug("Symbol " + name + " has conflicting fully qualified names:" + fqns);
        LOG.debug("It has been disambiguated with its latest Python version available symbol.");
      }
      return disambiguateWithLatestPythonSymbol(symbols);
    }
    return symbols.iterator().next();
  }

  private static boolean isBuiltinToDisambiguate(String moduleFqn, String name) {
    return moduleFqn.equals(BUILTINS_FQN) && BUILTINS_TO_DISAMBIGUATE.contains(name);
  }

  private static boolean haveAllTheSameFqn(Set<Symbol> symbols) {
    String firstFqn = symbols.iterator().next().fullyQualifiedName();
    return firstFqn != null && symbols.stream().map(Symbol::fullyQualifiedName).allMatch(firstFqn::equals);
  }

  private static AmbiguousSymbol fromOverloadedFunction(OverloadedFunctionSymbol overloadedFunctionSymbol, @Nullable String containerClassFqn, String moduleName) {
    Set<Symbol> overloadedSymbols = overloadedFunctionSymbol.getDefinitionsList().stream()
      .map(def -> new FunctionSymbolImpl(def, containerClassFqn, overloadedFunctionSymbol.getValidForList(), moduleName))
      .collect(Collectors.toSet());
    return AmbiguousSymbolImpl.create(overloadedSymbols);
  }
}
