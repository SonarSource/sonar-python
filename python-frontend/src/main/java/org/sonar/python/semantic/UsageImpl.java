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
package org.sonar.python.semantic;

import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Tree;

public class UsageImpl implements Usage {

  private final Tree tree;
  private final Kind kind;

  public UsageImpl(Tree tree, Kind kind) {
    this.tree = tree;
    this.kind = kind;
  }

  @Override
  public Tree tree() {
    return tree;
  }

  @Override
  public Kind kind() {
    return kind;
  }
}
