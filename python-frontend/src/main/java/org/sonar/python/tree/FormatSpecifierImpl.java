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
import org.sonar.plugins.python.api.tree.FormatSpecifier;
import org.sonar.plugins.python.api.tree.FormattedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class FormatSpecifierImpl extends PyTree implements FormatSpecifier {

  private Token columnToken;
  private List<Tree> fStringMiddles;

  public FormatSpecifierImpl(Token columnToken, List<Tree> fStringMiddles) {
    this.columnToken = columnToken;
    this.fStringMiddles = fStringMiddles;
  }

  @Override
  public Token columnToken() {
    return columnToken;
  }

  @Override
  public List<FormattedExpression> formatExpressions() {
    return fStringMiddles.stream()
      .filter(FormattedExpression.class::isInstance)
      .map(FormattedExpression.class::cast)
      .toList();
  }

  @Override
  List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>();
    children.add(columnToken);
    children.addAll(fStringMiddles);
    return children;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitFormatSpecifier(this);
  }

  @Override
  public Kind getKind() {
    return Kind.FORMAT_SPECIFIER;
  }
}
