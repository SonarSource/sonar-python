/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic.v2;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
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
  private static final Logger LOG = LoggerFactory.getLogger(TypeShed.class);

  private static final String PROTOBUF_BASE_RESOURCE_PATH = "/org/sonar/python/types/";
  private static final String PROTOBUF_CUSTOM_STUBS = PROTOBUF_BASE_RESOURCE_PATH + "custom_protobuf/";
  private static final String PROTOBUF = PROTOBUF_BASE_RESOURCE_PATH + "stdlib_protobuf/";
  private static final String PROTOBUF_THIRD_PARTY = PROTOBUF_BASE_RESOURCE_PATH + "third_party_protobuf/";
  private static final String PROTOBUF_THIRD_PARTY_MYPY = PROTOBUF_BASE_RESOURCE_PATH + "third_party_protobuf_mypy/";
  private static final String BUILTINS_FQN = "builtins";
  private static final String BUILTINS_PREFIX = BUILTINS_FQN + ".";
  // Those fundamentals builtins symbols need not to be ambiguous for the frontend to work properly
  private static final Set<String> BUILTINS_TO_DISAMBIGUATE = Stream.concat(
    Stream.of(INT, FLOAT, COMPLEX, STR, BuiltinTypes.SET, DICT, LIST, TUPLE, NONE_TYPE, BOOL, "type", "super", "frozenset", "memoryview"),
    BuiltinSymbols.EXCEPTIONS.stream()
  ).collect(Collectors.toSet());
  // This is needed for some Python 2 modules whose name differ from their Python 3 counterpart by capitalization only.
  private static final Map<String, String> MODULES_TO_DISAMBIGUATE = Map.of(
    "ConfigParser", "2@ConfigParser",
    "Queue", "2@Queue",
    "SocketServer", "2@SocketServer"
  );

  private Set<String> supportedPythonVersions;
  private Map<String, Symbol> builtins;
  private final Map<String, Map<String, Symbol>> typeShedSymbols;
  private final Set<String> modulesInProgress;
  private final ProjectLevelSymbolTable projectLevelSymbolTable;

  public TypeShed(ProjectLevelSymbolTable projectLevelSymbolTable) {
    // workaround to initialize supported python versions used in ClassSymbolImpl
    // TODO: remove once v2 types model will be populated from TypeShed bypassing conversion to symbols
    org.sonar.python.types.TypeShed.builtinSymbols();
    typeShedSymbols = new HashMap<>();
    modulesInProgress = new HashSet<>();
    this.projectLevelSymbolTable = projectLevelSymbolTable;
  }

  //================================================================================
  // Public methods
  //================================================================================

  public Map<String, Symbol> builtinSymbols() {
    if (builtins == null) {
      supportedPythonVersions();
      Map<String, Symbol> symbols = getSymbolsFromProtobufModule(BUILTINS_FQN, PROTOBUF);
      symbols.put(NONE_TYPE, new ClassSymbolImpl(NONE_TYPE, NONE_TYPE));
      builtins = Collections.unmodifiableMap(symbols);
    }
    return builtins;
  }

  private Set<String> supportedPythonVersions() {
    if (supportedPythonVersions == null) {
      supportedPythonVersions =
        ProjectPythonVersion.currentVersions().stream().map(PythonVersionUtils.Version::serializedValue).collect(Collectors.toSet());
    }
    return supportedPythonVersions;
  }

  private boolean searchedModuleMatchesCurrentProject(String searchedModule) {
    return projectLevelSymbolTable.projectBasePackages().contains(searchedModule.split("\\.", 2)[0]);
  }

  /**
   * Returns map of exported symbols by name for a given module
   */
  public Map<String, Symbol> symbolsForModule(String moduleName) {
    if (searchedModuleMatchesCurrentProject(moduleName)) {
      return Collections.emptyMap();
    }
    if (!typeShedSymbols.containsKey(moduleName)) {
      var symbols = Optional.of(org.sonar.python.types.TypeShed.getLoadedTypeShedSymbols())
        .filter(m -> m.containsKey(moduleName))
        .map(m -> m.get(moduleName))
        .orElseGet(() -> searchTypeShedForModule(moduleName));

      typeShedSymbols.put(moduleName, symbols);
      return symbols;
    }
    return typeShedSymbols.get(moduleName);
  }

  @CheckForNull
  public Symbol symbolWithFQN(String fullyQualifiedName) {
    Map<String, Symbol> builtinSymbols = builtinSymbols();
    Symbol builtinSymbol = builtinSymbols.get(normalizedFqn(fullyQualifiedName));
    if (builtinSymbol != null) {
      return builtinSymbol;
    }
    String[] fqnSplittedByDot = fullyQualifiedName.split("\\.");
    String moduleName = Arrays.stream(fqnSplittedByDot, 0, fqnSplittedByDot.length - 1).collect(Collectors.joining("."));
    return symbolWithFQN(moduleName, fullyQualifiedName);
  }

  @CheckForNull
  protected Symbol symbolWithFQN(String stdLibModuleName, String fullyQualifiedName) {
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

  private static String normalizedFqn(String fqn) {
    if (fqn.startsWith(BUILTINS_PREFIX)) {
      return fqn.substring(BUILTINS_PREFIX.length());
    }
    return fqn;
  }

  private boolean isValidForProjectPythonVersion(List<String> validForPythonVersions) {
    if (validForPythonVersions.isEmpty()) {
      return true;
    }
    // TODO: SONARPY-1522 - remove this workaround when we will have all the stubs for Python 3.12.
    if (supportedPythonVersions().stream().allMatch(PythonVersionUtils.Version.V_312.serializedValue()::equals)
      && validForPythonVersions.contains(PythonVersionUtils.Version.V_311.serializedValue())) {
      return true;
    }
    HashSet<String> intersection = new HashSet<>(validForPythonVersions);
    intersection.retainAll(supportedPythonVersions());
    return !intersection.isEmpty();
  }

  private Set<Symbol> symbolsFromProtobufDescriptors(Set<Object> protobufDescriptors, String moduleName) {
    Set<Symbol> symbols = new HashSet<>();
    for (Object descriptor : protobufDescriptors) {
      if (descriptor instanceof SymbolsProtos.ClassSymbol classSymbolProto) {
        symbols.add(new ClassSymbolImpl(classSymbolProto, moduleName));
      }
      if (descriptor instanceof SymbolsProtos.FunctionSymbol functionSymbolProto) {
        symbols.add(new FunctionSymbolImpl(functionSymbolProto, null, moduleName));
      }
      if (descriptor instanceof OverloadedFunctionSymbol overloadedFunctionSymbol) {
        if (overloadedFunctionSymbol.getDefinitionsList().size() < 2) {
          throw new IllegalStateException("Overloaded function symbols should have at least two definitions.");
        }
        symbols.add(fromOverloadedFunction(((OverloadedFunctionSymbol) descriptor), moduleName));
      }
      if (descriptor instanceof SymbolsProtos.VarSymbol varSymbol) {
        SymbolImpl symbol = new SymbolImpl(varSymbol, moduleName, false);
        if (varSymbol.getIsImportedModule()) {
          Map<String, Symbol> moduleExportedSymbols = symbolsForModule(varSymbol.getFullyQualifiedName());
          moduleExportedSymbols.values().forEach(symbol::addChildSymbol);
        }
        symbols.add(symbol);
      }
    }
    return symbols;
  }

  //================================================================================
  // Private methods
  //================================================================================
  private Map<String, Symbol> searchTypeShedForModule(String moduleName) {
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
  private static Symbol disambiguateWithLatestPythonSymbol(Set<Symbol> alternatives) {
    int max = Integer.MIN_VALUE;
    Symbol latestPythonSymbol = null;
    for (Symbol alternative : alternatives) {
      int maxPythonVersionForSymbol =
        ((SymbolImpl) alternative).validForPythonVersions().stream().mapToInt(Integer::parseInt).max().orElse(max);
      if (maxPythonVersionForSymbol > max) {
        max = maxPythonVersionForSymbol;
        latestPythonSymbol = alternative;
      }
    }
    return latestPythonSymbol;
  }

  private Map<String, Symbol> getSymbolsFromProtobufModule(String moduleName, String dirName) {
    String fileName = MODULES_TO_DISAMBIGUATE.getOrDefault(moduleName, moduleName);
    InputStream resource = this.getClass().getResourceAsStream(dirName + fileName + ".protobuf");
    if (resource == null) {
      return Collections.emptyMap();
    }
    return getSymbolsFromProtobufModule(deserializedModule(moduleName, resource));
  }

  Map<String, Symbol> getSymbolsFromProtobufModule(@Nullable ModuleSymbol moduleSymbol) {
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
      Set<Symbol> symbols = symbolsFromProtobufDescriptors(entry.getValue(), moduleSymbol.getFullyQualifiedName());
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
        LOG.debug("Symbol {} has conflicting fully qualified names: {}", name, fqns);
        LOG.debug("It has been disambiguated with its latest Python version available symbol.");
      }
      return disambiguateWithLatestPythonSymbol(symbols);
    }
    return symbols.iterator().next();
  }

  @CheckForNull
  static ModuleSymbol deserializedModule(String moduleName, InputStream resource) {
    try {
      return ModuleSymbol.parseFrom(resource);
    } catch (IOException e) {
      LOG.debug("Error while deserializing protobuf for module {}", moduleName, e);
      return null;
    }
  }

  private static boolean isBuiltinToDisambiguate(String moduleFqn, String name) {
    return moduleFqn.equals(BUILTINS_FQN) && BUILTINS_TO_DISAMBIGUATE.contains(name);
  }

  private static boolean haveAllTheSameFqn(Set<Symbol> symbols) {
    String firstFqn = symbols.iterator().next().fullyQualifiedName();
    return firstFqn != null && symbols.stream().map(Symbol::fullyQualifiedName).allMatch(firstFqn::equals);
  }

  private static AmbiguousSymbol fromOverloadedFunction(OverloadedFunctionSymbol overloadedFunctionSymbol, String moduleName) {
    Set<Symbol> overloadedSymbols = overloadedFunctionSymbol.getDefinitionsList().stream()
      .map(def -> new FunctionSymbolImpl(def, null, overloadedFunctionSymbol.getValidForList(), moduleName))
      .collect(Collectors.toSet());
    return AmbiguousSymbolImpl.create(overloadedSymbols);
  }
}
