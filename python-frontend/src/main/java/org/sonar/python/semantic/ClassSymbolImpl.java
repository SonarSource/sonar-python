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


import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;

public class ClassSymbolImpl extends SymbolImpl implements ClassSymbol {

  private List<Symbol> superClasses = new ArrayList<>();
  private boolean hasSuperClassWithoutSymbol = false;
  private final Set<Symbol> members = new HashSet<>();
  private boolean hasAlreadyReadSuperClasses = false;

  public ClassSymbolImpl(String name, @Nullable String fullyQualifiedName) {
    super(name, fullyQualifiedName);
    this.setKind(Kind.CLASS);
  }

  @Override
  ClassSymbolImpl copyWithoutUsages() {
    ClassSymbolImpl copiedClassSymbol = new ClassSymbolImpl(name(), fullyQualifiedName());
    for (Symbol superClass : superClasses()) {
      if (superClass.kind() == Symbol.Kind.CLASS) {
        copiedClassSymbol.superClasses.add(((ClassSymbolImpl) superClass).copyWithoutUsages());
      } else {
        copiedClassSymbol.superClasses.add(new SymbolImpl(superClass.name(), superClass.fullyQualifiedName()));
      }
    }
    copiedClassSymbol.addMembers(members.stream().map(m -> ((SymbolImpl) m).copyWithoutUsages()).collect(Collectors.toList()));
    if (hasSuperClassWithoutSymbol) {
      copiedClassSymbol.setHasSuperClassWithoutSymbol();
    }
    return copiedClassSymbol;
  }

  @Override
  public List<Symbol> superClasses() {
    hasAlreadyReadSuperClasses = true;
    return Collections.unmodifiableList(superClasses);
  }

  public void addSuperClass(Symbol symbol) {
    if (hasAlreadyReadSuperClasses) {
      throw new IllegalStateException("Cannot call addSuperClass, super classes were already read");
    }
    this.superClasses.add(symbol);
  }

  @Override
  public boolean hasUnresolvedTypeHierarchy() {
    for (ClassSymbol superClass : allSuperClasses()) {
      if (superClass.superClasses().stream().anyMatch(s -> s.kind() != Kind.CLASS) || ((ClassSymbolImpl) superClass).hasSuperClassWithoutSymbol) {
        return true;
      }
    }
    return false;
  }

  @Override
  public Set<Symbol> declaredMembers() {
    return members;
  }

  public void addMembers(Collection<Symbol> members) {
    this.members.addAll(members);
    members.stream()
      .filter(m -> m.kind() == Kind.FUNCTION)
      .forEach(m -> ((FunctionSymbolImpl) m).setOwner(this));
  }

  public void setHasSuperClassWithoutSymbol() {
    this.hasSuperClassWithoutSymbol = true;
  }

  private Set<ClassSymbol> allSuperClasses() {
    Set<ClassSymbol> set = new HashSet<>();
    exploreSuperClasses(this, set);
    return set;
  }

  private static void exploreSuperClasses(ClassSymbol classSymbol, Set<ClassSymbol> set) {
    if (set.add(classSymbol)) {
      for (Symbol superClass : classSymbol.superClasses()) {
        if (superClass instanceof ClassSymbol) {
          exploreSuperClasses((ClassSymbol) superClass, set);
        }
      }
    }
  }
}
