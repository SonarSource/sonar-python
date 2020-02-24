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
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;

public class ClassSymbolImpl extends SymbolImpl implements ClassSymbol {

  private List<Symbol> superClasses = new ArrayList<>();
  private boolean hasUnresolvedTypeHierarchy = true;

  public ClassSymbolImpl(String name, @Nullable String fullyQualifiedName) {
    super(name, fullyQualifiedName);
    this.setKind(Kind.CLASS);
  }

  ClassSymbolImpl copyWithoutUsages() {
    ClassSymbolImpl copiedClassSymbol = new ClassSymbolImpl(name(), fullyQualifiedName());
    for (Symbol superClass : superClasses()) {
      if (superClass.kind() == Symbol.Kind.CLASS) {
        copiedClassSymbol.superClasses.add(((ClassSymbolImpl) superClass).copyWithoutUsages());
      } else {
        copiedClassSymbol.superClasses.add(new SymbolImpl(superClass.name(), superClass.fullyQualifiedName()));
      }
    }
    return copiedClassSymbol;
  }

  @Override
  public List<Symbol> superClasses() {
    return Collections.unmodifiableList(superClasses);
  }

  public void addSuperClass(Symbol symbol) {
    this.superClasses.add(symbol);
  }

  @Override
  public boolean hasUnresolvedTypeHierarchy() {
    return hasUnresolvedTypeHierarchy;
  }

  public void setHasUnresolvedTypeHierarchy(boolean hasUnresolvedTypeHierarchy) {
    this.hasUnresolvedTypeHierarchy = hasUnresolvedTypeHierarchy;
  }
}
