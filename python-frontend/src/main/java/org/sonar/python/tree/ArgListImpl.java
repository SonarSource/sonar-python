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

import java.util.ArrayList;
import java.util.List;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ArgListImpl extends PyTree implements ArgList {

  private final List<Argument> arguments;
  private final List<Token> commas;

  public ArgListImpl(List<Argument> arguments, List<Token> commas) {
    this.arguments = arguments;
    this.commas = commas;
  }

  @Override
  public List<Argument> arguments() {
    return arguments;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitArgumentList(this);
  }

  @Override
  public List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>();
    int i = 0;
    for (Tree argument : arguments) {
      children.add(argument);
      if (i < commas.size()) {
        children.add(commas.get(i));
      }
      i++;
    }
    return children;
  }

  @Override
  public Kind getKind() {
    return Tree.Kind.ARG_LIST;
  }
}
