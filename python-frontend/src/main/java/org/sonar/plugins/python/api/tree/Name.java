/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python.api.tree;

import javax.annotation.CheckForNull;
import org.sonar.api.Beta;
import org.sonar.python.semantic.v2.SymbolV2;

/**
 * See https://docs.python.org/3/reference/expressions.html#atom-identifiers
 */
public interface Name extends Expression, HasSymbol {

  String name();

  // FIXME: we should create a separate tree for Variables
  boolean isVariable();

  @Beta
  @CheckForNull
  SymbolV2 symbolV2();
}
