/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.TypeParam;
import org.sonar.plugins.python.api.tree.TypeParams;

public class TypeParamsImpl extends PyTree implements TypeParams {

  private final Token leftPar;
  private final List<TypeParam> typeParamsList;
  private final List<Token> commas;
  private final Token rightPar;

  public TypeParamsImpl(Token leftPar, List<TypeParam> typeParamsList, List<Token> commas, Token rightPar) {
    this.leftPar = leftPar;
    this.typeParamsList = typeParamsList;
    this.commas = commas;
    this.rightPar = rightPar;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitTypeParams(this);
  }

  @Override
  public Kind getKind() {
    return Kind.TYPE_PARAMS;
  }

  @Override
  public Token leftBracket() {
    return leftPar;
  }

  @Override
  public List<TypeParam> typeParamsList() {
    return typeParamsList;
  }

  @Override
  public Token rightBracket() {
    return rightPar;
  }

  @Override
  List<Tree> computeChildren() {
    var builder = Stream.<Tree>builder();
    builder.add(leftPar);
    typeParamsList.forEach(builder::add);
    commas.forEach(builder::add);
    builder.add(rightPar);

    return builder.build()
      .filter(Objects::nonNull)
      .sorted(Comparator.<Tree, Integer>comparing(t -> t.firstToken().line()).thenComparing(t -> t.firstToken().column()))
      .collect(Collectors.toList());
  }
}
