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

import com.sonar.sslr.api.AstNode;
import org.sonar.python.api.tree.Token;
import java.util.Collections;
import java.util.List;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.SetLiteral;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class SetLiteralImpl extends DictOrSetLiteralImpl implements SetLiteral {
  private final List<Expression> elements;

  public SetLiteralImpl(AstNode node, Token lCurlyBrace, List<Expression> elements, List<Token> commas, Token rCurlyBrace) {
    super(node, lCurlyBrace, commas, rCurlyBrace);
    this.elements = elements;
  }

  @Override
  public List<Expression> elements() {
    return elements;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitSetLiteral(this);
  }

  @Override
  public List<Tree> children() {
    return Collections.unmodifiableList(elements);
  }

  @Override
  public Kind getKind() {
    return Kind.SET_LITERAL;
  }
}
