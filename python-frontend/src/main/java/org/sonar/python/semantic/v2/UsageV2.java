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
package org.sonar.python.semantic.v2;

import org.sonar.api.Beta;
import org.sonar.plugins.python.api.tree.Tree;

@Beta
public record UsageV2(Tree tree, Kind kind) {

  @Beta
  public boolean isBindingUsage() {
    return kind() != UsageV2.Kind.OTHER && kind() != UsageV2.Kind.GLOBAL_DECLARATION;
  }

  public enum Kind {
    ASSIGNMENT_LHS,
    COMPOUND_ASSIGNMENT_LHS,
    IMPORT,
    LOOP_DECLARATION,
    COMP_DECLARATION,
    OTHER,
    PARAMETER,
    FUNC_DECLARATION,
    CLASS_DECLARATION,
    EXCEPTION_INSTANCE,
    WITH_INSTANCE,
    GLOBAL_DECLARATION,
    PATTERN_DECLARATION,
    TYPE_PARAM_DECLARATION,
    TYPE_ALIAS_DECLARATION,
  }
}
