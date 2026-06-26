/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.api.nosonar;

import java.util.Set;

public record NoSonarLineInfo(Set<String> suppressedRuleKeys, String comment, boolean securityOnlySuppression) {
  public NoSonarLineInfo(Set<String> suppressedRuleKeys) {
    this(suppressedRuleKeys, "", false);
  }

  public NoSonarLineInfo(Set<String> suppressedRuleKeys, String comment) {
    this(suppressedRuleKeys, comment, false);
  }

  public static NoSonarLineInfo securityOnly(String comment) {
    return new NoSonarLineInfo(Set.of(), comment, true);
  }

  public static NoSonarLineInfo securityOnly() {
    return securityOnly("");
  }

  public boolean isSuppressedRuleKeysEmpty() {
    return suppressedRuleKeys.isEmpty();
  }
}
