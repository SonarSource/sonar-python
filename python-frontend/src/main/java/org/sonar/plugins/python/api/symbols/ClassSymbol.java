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
package org.sonar.plugins.python.api.symbols;

import com.google.common.annotations.Beta;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.LocationInFile;

public interface ClassSymbol extends Symbol {
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
}
