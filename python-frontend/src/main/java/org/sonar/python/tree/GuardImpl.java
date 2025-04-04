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
package org.sonar.python.tree;

import java.util.Arrays;
import java.util.List;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Guard;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class GuardImpl extends PyTree implements Guard {
  private final Token ifKeyword;
  private final Expression condition;

  public GuardImpl(Token ifKeyword, Expression condition) {
    this.ifKeyword = ifKeyword;
    this.condition = condition;
  }

  @Override
  public Token ifKeyword() {
    return ifKeyword;
  }

  @Override
  public Expression condition() {
    return condition;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitGuard(this);
  }

  @Override
  public Kind getKind() {
    return Kind.GUARD;
  }

  @Override
  List<Tree> computeChildren() {
    return Arrays.asList(ifKeyword, condition);
  }
}
