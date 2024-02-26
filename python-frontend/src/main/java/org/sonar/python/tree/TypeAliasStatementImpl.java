/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.TypeAliasStatement;
import org.sonar.plugins.python.api.tree.TypeParams;

public class TypeAliasStatementImpl extends SimpleStatement implements TypeAliasStatement {
  private final Token typeKeyword;
  private final Name name;
  private final TypeParams typeParams;
  private final Token equalToken;
  private final Expression expression;
  private final Separators separator;

  public TypeAliasStatementImpl(Token typeKeyword, Name name, @Nullable TypeParams typeParams,
    Token equalToken, Expression expression, Separators separator) {
    this.typeKeyword = typeKeyword;
    this.name = name;
    this.typeParams = typeParams;
    this.equalToken = equalToken;
    this.expression = expression;
    this.separator = separator;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitTypeAliasStatement(this);
  }

  @Override
  public Kind getKind() {
    return Kind.TYPE_ALIAS_STMT;
  }

  @Override
  public Token typeKeyword() {
    return typeKeyword;
  }

  @Override
  public Name name() {
    return name;
  }

  @CheckForNull
  @Override
  public TypeParams typeParams() {
    return typeParams;
  }

  @Override
  public Token equalToken() {
    return equalToken;
  }

  @Override
  public Expression expression() {
    return expression;
  }

  @Override
  List<Tree> computeChildren() {
    var builder = Stream.<Tree>builder()
      .add(typeKeyword())
      .add(name())
      .add(typeParams())
      .add(equalToken()).add(expression());

    separator.elements().forEach(builder::add);
    return builder.build().filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Token separator() {
    return separator.last();
  }
}
