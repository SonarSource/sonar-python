/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.symbols.Symbol;

@Rule(key = "S4487")
public class UnreadPrivateAttributesCheck extends AbstractUnreadPrivateMembersCheck {

  private static final boolean DEFAULT_ENABLE_SINGLE_UNDERSCORE_ISSUES = false;

  @RuleProperty(
    key = "enableSingleUnderscoreIssues",
    description = "Enable issues on unread attributes with a single underscore prefix",
    defaultValue = "" + DEFAULT_ENABLE_SINGLE_UNDERSCORE_ISSUES)
  public boolean enableSingleUnderscoreIssues = DEFAULT_ENABLE_SINGLE_UNDERSCORE_ISSUES;

  @Override
  String memberPrefix() {
    return enableSingleUnderscoreIssues ? "_" : "__";
  }

  @Override
  Symbol.Kind kind() {
    return Symbol.Kind.OTHER;
  }

  @Override
  String message(String memberName) {
    return "Remove this unread private attribute '" + memberName + "' or refactor the code to use its value.";
  }

  @Override
  String secondaryMessage() {
    return "Also modified here.";
  }
}
