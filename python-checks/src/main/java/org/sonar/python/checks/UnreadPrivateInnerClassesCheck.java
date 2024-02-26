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
package org.sonar.python.checks;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;

@Rule(key = "S3985")
public class UnreadPrivateInnerClassesCheck extends AbstractUnreadPrivateMembersCheck {
  @Override
  String memberPrefix() {
    return "_";
  }

  @Override
  protected boolean isException(Symbol symbol) {
    return Optional.of(symbol)
      .filter(ClassSymbol.class::isInstance)
      .map(ClassSymbol.class::cast)
      .filter(ClassSymbol::hasDecorators)
      .isPresent();
  }

  @Override
  Symbol.Kind kind() {
    return CLASS;
  }

  @Override
  String message(String memberName) {
    return "Remove this unused private '" + memberName + "' class.";
  }

  @Override
  String secondaryMessage() {
    return null;
  }
}
