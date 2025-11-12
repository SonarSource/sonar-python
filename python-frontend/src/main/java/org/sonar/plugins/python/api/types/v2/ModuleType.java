/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.plugins.python.api.types.v2;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.TriBool;

@Beta
public final class ModuleType implements PythonType {
  private final String name;
  private final String fullyQualifiedName;
  private final ModuleType parent;
  private final Map<String, TypeWrapper> members;
  private final Map<String, TypeWrapper> subModules;

  public ModuleType(@Nullable String name,
    @Nullable String fullyQualifiedName,
    @Nullable ModuleType parent,
    Map<String, TypeWrapper> members) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.parent = parent;
    this.members = members;
    this.subModules = new ConcurrentHashMap<>();
    registerAsSubmoduleOfParent(parent);
  }

  private void registerAsSubmoduleOfParent(@Nullable ModuleType parent) {
    if (parent == null) {
      return;
    }
    parent.subModules.putIfAbsent(this.name, TypeWrapper.of(this));
  }

  public ModuleType(@Nullable String name) {
    this(name, null);
  }

  public ModuleType(@Nullable String name, @Nullable ModuleType parent) {
    this(name, null, parent, new HashMap<>());
  }

  public ModuleType(@Nullable String name,
    @Nullable ModuleType parent,
    Map<String, TypeWrapper> members) {
    this(name, null, parent, members);
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return Optional.ofNullable(members.get(memberName)).map(TypeWrapper::type).or(() -> resolveSubmodule(memberName));
  }

  public Optional<PythonType> resolveSubmodule(String submoduleName) {
    return Optional.ofNullable(subModules.get(submoduleName)).map(TypeWrapper::type);
  }

  @Override
  public TriBool hasMember(String memberName) {
    if (resolveMember(memberName).isPresent()) {
      return TriBool.TRUE;
    }
    return TriBool.UNKNOWN;
  }

  @Override
  public String toString() {
    return "ModuleType{" +
      "name='" + name + '\'' +
      ", members=" + members +
      '}';
  }

  @Override
  @CheckForNull
  public String name() {
    return name;
  }

  @CheckForNull
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  @CheckForNull
  public ModuleType parent() {
    return parent;
  }

  public Map<String, TypeWrapper> members() {
    return members;
  }

}
