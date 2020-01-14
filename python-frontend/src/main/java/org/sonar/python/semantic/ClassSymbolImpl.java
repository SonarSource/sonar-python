/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

  List<Symbol> parents = new ArrayList<>();
  boolean hasUnresolvedParents = true;

  public ClassSymbolImpl(String name, @Nullable String fullyQualifiedName) {
    super(name, fullyQualifiedName);
    this.setKind(Kind.CLASS);
  }

  @Override
  public List<Symbol> parents() {
    return Collections.unmodifiableList(parents);
  }

  public void addParent(Symbol symbol) {
    this.parents.add(symbol);
  }

  @Override
  public boolean hasUnresolvedParents() {
    return hasUnresolvedParents;
  }
}
