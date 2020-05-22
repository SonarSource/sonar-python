/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import org.sonar.plugins.python.api.symbols.ModuleSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import javax.annotation.Nullable;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class ModuleSymbolImpl extends SymbolImpl implements ModuleSymbol {
  private final Set<Symbol> members = new HashSet<>();

  public ModuleSymbolImpl(String name, @Nullable String fullyQualifiedName) {
    super(name, fullyQualifiedName);
    this.setKind(Kind.MODULE);
  }

  @Override
  public Set<Symbol> declaredMembers() {
    return members;
  }

  @Override
  ModuleSymbolImpl copyWithoutUsages() {
    ModuleSymbolImpl copiedModuleSymbol = new ModuleSymbolImpl(name(), fullyQualifiedName());
    copiedModuleSymbol.addMembers(members.stream().map(m -> ((SymbolImpl) m).copyWithoutUsages()).collect(Collectors.toList()));
    return copiedModuleSymbol;
  }

  public void addMembers(Collection<Symbol> members) {
    this.members.addAll(members);
  }

  public void addMember(Symbol member) {
    this.members.add(member);
  }
}
