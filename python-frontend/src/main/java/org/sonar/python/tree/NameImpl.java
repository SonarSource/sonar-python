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

import java.util.Collections;
import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

public class NameImpl extends PyTree implements Name {
  private final Token token;
  private final String name;
  private final boolean isVariable;
  private Symbol symbol;
  private Usage usage;
  private InferredType inferredType = InferredTypes.anyType();
  private static final String TRUE = "True";
  private static final String FALSE = "False";

  public NameImpl(Token token, boolean isVariable) {
    this.token = token;
    this.name = token.value();
    this.isVariable = isVariable;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public boolean isVariable() {
    boolean isKeyword = false;
    if (parent() != null && parent().is(Kind.REGULAR_ARGUMENT)) {
      isKeyword = ((RegularArgument) parent()).keywordArgument() == this;
    }
    return isVariable && !isKeyword;
  }

  @Override
  public Kind getKind() {
    return Kind.NAME;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitName(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Collections.singletonList(token);
  }

  @CheckForNull
  @Override
  public Symbol symbol() {
    return symbol;
  }

  @CheckForNull
  @Override
  public Usage usage() {
    return usage;
  }

  public void setSymbol(Symbol symbol) {
    this.symbol = symbol;
  }

  public void setUsage(Usage usage) {
    this.usage = usage;
  }

  @Override
  public InferredType type() {
    if (symbol != null) {
      if (isBooleanBuiltinSymbol()) {
        return InferredTypes.BOOL;
      }
      if (symbol.kind() == Symbol.Kind.CLASS) {
        return InferredTypes.TYPE;
      }
    }
    return inferredType;
  }

  private boolean isBooleanBuiltinSymbol() {
    return (TRUE.equals(symbol.name()) && TRUE.equals(symbol.fullyQualifiedName())) ||
      (FALSE.equals(symbol.name()) && FALSE.equals(symbol.fullyQualifiedName()));
  }

  public void setInferredType(InferredType inferredType) {
    this.inferredType = inferredType;
  }
}
