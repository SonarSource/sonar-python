/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
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
