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
package org.sonar.python.semantic.v2.types;

import java.util.Set;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

public class Definition extends Propagation {

  final SymbolV2 lhsSymbol;
  Name lhsName;

  public Definition(SymbolV2 symbol, Name name) {
    super();
    this.lhsSymbol = symbol;
    this.lhsName = name;
  }

  @Override
  boolean propagate(Set<SymbolV2> initializedVars) {
    PythonType rhsType = lhsName.typeV2();
    if (initializedVars.add(lhsSymbol)) {
      lhsSymbol.usages().stream().map(UsageV2::tree).filter(NameImpl.class::isInstance).map(NameImpl.class::cast).forEach(n -> n.typeV2(rhsType));
      return true;
    } else {
      PythonType currentType = lhsName.typeV2();
      PythonType newType = UnionType.or(rhsType, currentType);
      lhsSymbol.usages().stream().map(UsageV2::tree).filter(NameImpl.class::isInstance).map(NameImpl.class::cast).forEach(n -> n.typeV2(newType));
      return !newType.equals(currentType);
    }
  }

  public SymbolV2 lhsSymbol() {
    return lhsSymbol;
  }

  public Name lhsName() {
    return lhsName;
  }
}
