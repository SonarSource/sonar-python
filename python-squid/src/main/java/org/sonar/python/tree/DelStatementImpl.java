/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.tree;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.DelStatement;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TreeVisitor;

public class DelStatementImpl extends PyTree implements DelStatement {
  private final Token delKeyword;
  private final List<Expression> expressionTrees;
  private final Token separator;

  public DelStatementImpl(Token delKeyword, List<Expression> expressionTrees, @Nullable Token separator) {
    super(delKeyword, expressionTrees.isEmpty() ? delKeyword : expressionTrees.get(expressionTrees.size() - 1).lastToken());
    this.delKeyword = delKeyword;
    this.expressionTrees = expressionTrees;
    this.separator = separator;
  }

  @Override
  public Token delKeyword() {
    return delKeyword;
  }

  @Override
  public List<Expression> expressions() {
    return expressionTrees;
  }

  @Nullable
  @Override
  public Token separator() {
    return separator;
  }

  @Override
  public Kind getKind() {
    return Kind.DEL_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitDelStatement(this);
  }

  @Override
  public List<Tree> childs() {
    return Stream.of(Collections.singletonList(delKeyword), expressionTrees, Collections.singletonList(separator))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
