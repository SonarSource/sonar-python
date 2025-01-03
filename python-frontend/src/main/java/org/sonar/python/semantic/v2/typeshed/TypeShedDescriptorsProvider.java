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
package org.sonar.python.semantic.v2.typeshed;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Predicate;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.ModuleDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos.ModuleSymbol;

import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

public class TypeShedDescriptorsProvider {
  private static final Logger LOG = LoggerFactory.getLogger(TypeShedDescriptorsProvider.class);

  private static final String PROTOBUF_BASE_RESOURCE_PATH = "/org/sonar/python/types/";
  private static final String PROTOBUF_CUSTOM_STUBS = PROTOBUF_BASE_RESOURCE_PATH + "custom_protobuf/";
  private static final String PROTOBUF = PROTOBUF_BASE_RESOURCE_PATH + "stdlib_protobuf/";
  private static final String PROTOBUF_THIRD_PARTY = PROTOBUF_BASE_RESOURCE_PATH + "third_party_protobuf/";
  private static final String PROTOBUF_THIRD_PARTY_MYPY = PROTOBUF_BASE_RESOURCE_PATH + "third_party_protobuf_mypy/";
  public static final String BUILTINS_FQN = "builtins";
  // This is needed for some Python 2 modules whose name differ from their Python 3 counterpart by capitalization only.
  private static final Map<String, String> MODULES_TO_DISAMBIGUATE = Map.of(
    "ConfigParser", "2@ConfigParser",
    "Queue", "2@Queue",
    "SocketServer", "2@SocketServer"
  );
  private final ModuleSymbolToDescriptorConverter moduleConverter;

  private Map<String, Descriptor> builtins;
  private final Set<String> projectBasePackages;
  private final Map<String, Map<String, Descriptor>> cachedDescriptors;

  public TypeShedDescriptorsProvider(Set<String> projectBasePackages) {
    this(projectBasePackages, ProjectPythonVersion.currentVersions());
  }

  public TypeShedDescriptorsProvider(Set<String> projectBasePackages, Set<PythonVersionUtils.Version> projectPythonVersions) {
    moduleConverter = new ModuleSymbolToDescriptorConverter(projectPythonVersions);
    cachedDescriptors = new HashMap<>();
    this.projectBasePackages = projectBasePackages;
  }

  //================================================================================
  // Public methods
  //================================================================================

  public Map<String, Descriptor> builtinDescriptors() {
    if (builtins == null) {
      Map<String, Descriptor> symbols = getModuleDescriptors(BUILTINS_FQN, PROTOBUF);
      symbols.put(NONE_TYPE, new ClassDescriptor.ClassDescriptorBuilder().withName(NONE_TYPE).withFullyQualifiedName(NONE_TYPE).build());
      builtins = Collections.unmodifiableMap(symbols);
    }
    return builtins;
  }

  /**
   * Returns map of exported symbols by name for a given module
   */
  public Map<String, Descriptor> descriptorsForModule(String moduleName) {
    if (searchedModuleMatchesCurrentProject(moduleName)) {
      return Collections.emptyMap();
    }
    return cachedDescriptors.computeIfAbsent(moduleName, this::searchTypeShedForModule);
  }

  public Set<String> stubModules() {
    return cachedDescriptors.keySet();
  }

  //================================================================================
  // Private methods
  //================================================================================

  private boolean searchedModuleMatchesCurrentProject(String searchedModule) {
    return projectBasePackages.contains(searchedModule.split("\\.", 2)[0]);
  }

  private Map<String, Descriptor> searchTypeShedForModule(String moduleName) {
    return Stream.of(PROTOBUF_CUSTOM_STUBS, PROTOBUF, PROTOBUF_THIRD_PARTY_MYPY)
      .map(dirName -> getModuleDescriptors(moduleName, dirName))
      .filter(Predicate.not(Map::isEmpty))
      .findFirst()
      .orElseGet(() -> getModuleDescriptors(moduleName, PROTOBUF_THIRD_PARTY));
  }

  private Map<String, Descriptor> getModuleDescriptors(String moduleName, String dirName) {
    String fileName = MODULES_TO_DISAMBIGUATE.getOrDefault(moduleName, moduleName);
    InputStream resource = this.getClass().getResourceAsStream(dirName + fileName + ".protobuf");
    if (resource == null) {
      return Collections.emptyMap();
    }
    var moduleSymbol = deserializedModule(moduleName, resource);
    var moduleDescriptor = moduleConverter.convert(moduleSymbol);
    return Optional.ofNullable(moduleDescriptor).map(ModuleDescriptor::members).orElseGet(Map::of);
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

  public List<Descriptor> stubFilesDescriptors() {
    List<Descriptor> descriptors = new ArrayList<>(new TreeMap<>(builtinDescriptors()).values());
    new TreeMap<>(cachedDescriptors).values().forEach(entry -> descriptors.addAll(new TreeMap<>(entry).values()));
    return descriptors;
  }
}
