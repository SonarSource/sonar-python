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

import java.util.Collection;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.BuiltinSymbols;

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
  protected boolean isException(Symbol symbol) {
    return Optional.of(symbol)
      .filter(FunctionSymbol.class::isInstance)
      .map(FunctionSymbol.class::cast)
      .map(FunctionSymbol::decorators)
      .stream()
      .flatMap(Collection::stream)
      .anyMatch(decorator -> !BuiltinSymbols.STATIC_AND_CLASS_METHOD_DECORATORS.contains(decorator));
  }
}
