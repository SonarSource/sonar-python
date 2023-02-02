/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks;

import java.util.Objects;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;

@Rule(key = "S1144")
public class UnreadPrivateMethodsCheck extends AbstractUnreadPrivateMembersCheck {
  @Override
  String memberPrefix() {
    return "__";
  }

  @Override
  Symbol.Kind kind() {
    return FUNCTION;
  }

  @Override
  String message(String memberName) {
    return "Remove this unused class-private '" + memberName + "' method.";
  }

  @Override
  String secondaryMessage() {
    return null;
  }

  @Override
  protected boolean filterMember(Symbol symbol, Set<FunctionSymbol> decoratedMethods) {
    // check only methods if there is no other methods with decorators or there is other decorated methods
    return decoratedMethods.isEmpty() || decoratedMethods.stream().anyMatch(m -> !Objects.equals(symbol, m));
  }
}
