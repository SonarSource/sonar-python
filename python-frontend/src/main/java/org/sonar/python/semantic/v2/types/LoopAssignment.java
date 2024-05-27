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

import java.util.Collection;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

public class LoopAssignment extends Assignment {
  public LoopAssignment(SymbolV2 lhsSymbol, Name lhsName, Expression rhs, Map<SymbolV2, Set<Propagation>> propagationsByLhs) {
    super(lhsSymbol, lhsName, rhs, propagationsByLhs);
  }

  @Override
  public PythonType rhsType() {
    return Optional.of(super.rhsType())
      .filter(ObjectType.class::isInstance)
      .map(ObjectType.class::cast)
      .filter(t -> t.hasMember("__iter__") == TriBool.TRUE)
      .map(ObjectType::attributes)
      .map(Collection::stream)
      .flatMap(Stream::findFirst)
      .orElse(PythonType.UNKNOWN);
  }
}
