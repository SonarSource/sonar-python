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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;

public class ClassSymbolImpl extends SymbolImpl implements ClassSymbol {

  private final List<Symbol> superClasses = new ArrayList<>();
  private Set<Symbol> allSuperClasses = null;
  private boolean hasSuperClassWithoutSymbol = false;
  private final Set<Symbol> members = new HashSet<>();
  private Map<String, Symbol> membersByName = null;
  private boolean hasAlreadyReadSuperClasses = false;
  private boolean hasAlreadyReadMembers = false;

  public ClassSymbolImpl(String name, @Nullable String fullyQualifiedName) {
    super(name, fullyQualifiedName);
    this.setKind(Kind.CLASS);
  }

  @Override
  ClassSymbolImpl copyWithoutUsages() {
    ClassSymbolImpl copiedClassSymbol = new ClassSymbolImpl(name(), fullyQualifiedName());
    for (Symbol superClass : superClasses()) {
      if (superClass == this) {
        copiedClassSymbol.superClasses.add(copiedClassSymbol);
      } else if (superClass.kind() == Symbol.Kind.CLASS) {
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
    for (Symbol superClassSymbol : allSuperClasses()) {
      if (superClassSymbol.kind() != Kind.CLASS) {
        return true;
      }

      ClassSymbolImpl superClass = (ClassSymbolImpl) superClassSymbol;
      if (superClass.hasSuperClassWithoutSymbol) {
        return true;
      }
    }
    return false;
  }

  @Override
  public Set<Symbol> declaredMembers() {
    hasAlreadyReadMembers = true;
    return members;
  }

  @Override
  public Optional<Symbol> resolveMember(String memberName) {
    for (Symbol symbol : allSuperClasses()) {
      if (symbol.kind() == Kind.CLASS) {
        ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
        Symbol matchingMember = classSymbol.membersByName().get(memberName);
        if (matchingMember != null) {
          return Optional.of(matchingMember);
        }
      }
    }
    return Optional.empty();
  }

  @Override
  public boolean isOrExtends(String fullyQualifiedClassName) {
    return allSuperClasses().stream().anyMatch(c -> c.fullyQualifiedName() != null && Objects.equals(fullyQualifiedClassName, c.fullyQualifiedName()));
  }

  @Override
  public boolean isOrExtends(ClassSymbol other) {
    // TODO there should be only 1 class with a given fullyQualifiedName when analyzing a python file
    return allSuperClasses().stream().anyMatch(c -> Objects.equals(c.fullyQualifiedName(), other.fullyQualifiedName()));
  }

  private Map<String, Symbol> membersByName() {
    if (membersByName == null) {
      membersByName = declaredMembers().stream().collect(Collectors.toMap(Symbol::name, m -> m, (s1, s2) -> s1));
    }
    return membersByName;
  }

  public void addMembers(Collection<Symbol> members) {
    if (hasAlreadyReadMembers) {
      throw new IllegalStateException("Cannot call addMembers, members were already read");
    }
    this.members.addAll(members);
    members.stream()
      .filter(m -> m.kind() == Kind.FUNCTION)
      .forEach(m -> ((FunctionSymbolImpl) m).setOwner(this));
  }

  public void setHasSuperClassWithoutSymbol() {
    this.hasSuperClassWithoutSymbol = true;
  }

  private Set<Symbol> allSuperClasses() {
    if (allSuperClasses == null) {
      allSuperClasses = new LinkedHashSet<>();
      exploreSuperClasses(this, allSuperClasses);
    }
    return allSuperClasses;
  }

  private static void exploreSuperClasses(Symbol symbol, Set<Symbol> set) {
    if (set.add(symbol) && symbol.kind() == Kind.CLASS) {
      ClassSymbol classSymbol = (ClassSymbol) symbol;
      for (Symbol superClass : classSymbol.superClasses()) {
        exploreSuperClasses(superClass, set);
      }
    }
  }

  @Override
  public void removeUsages() {
    super.removeUsages();
    superClasses.forEach(symbol -> ((SymbolImpl) symbol).removeUsages());
    members.forEach(symbol -> ((SymbolImpl) symbol).removeUsages());
  }
}
