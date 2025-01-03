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
package org.sonar.plugins.python.api.symbols;

import com.google.common.annotations.Beta;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.LocationInFile;

public interface ClassSymbol extends Symbol {
  @Beta
  List<String> superClassesFqn();

  List<Symbol> superClasses();

  boolean hasUnresolvedTypeHierarchy();

  Set<Symbol> declaredMembers();

  @CheckForNull
  LocationInFile definitionLocation();

  @Beta
  Optional<Symbol> resolveMember(String memberName);

  @Beta
  boolean canHaveMember(String memberName);

  @Beta
  boolean isOrExtends(String fullyQualifiedClassName);

  @Beta
  boolean isOrExtends(ClassSymbol other);

  @Beta
  boolean canBeOrExtend(String fullyQualifiedClassName);

  @Beta
  boolean hasDecorators();

  @Beta
  boolean hasMetaClass();
}
