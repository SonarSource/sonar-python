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
import java.util.Collections;
import java.util.List;
import org.sonar.python.api.tree.PyNumericLiteralTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyNumericLiteralTreeImpl extends PyTree implements PyNumericLiteralTree {

  private final String valueAsString;

  PyNumericLiteralTreeImpl(AstNode node) {
    super(node);
    valueAsString = node.getTokenValue();
  }

  @Override
  public Kind getKind() {
    return Kind.NUMERIC_LITERAL;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitNumericLiteral(this);
  }

  @Override
  public long valueAsLong() {
    String literalValue = valueAsString.replace("_", "");
    if (literalValue.startsWith("0b") || literalValue.startsWith("0B")) {
      return Integer.valueOf(literalValue.substring(2), 2);
    }
    if (valueAsString.endsWith("L") || valueAsString.endsWith("l")) {
      literalValue = literalValue.substring(0, literalValue.length() - 1);
    }
    return Long.parseLong(literalValue);
  }

  @Override
  public String valueAsString() {
    return valueAsString;
  }

  @Override
  public List<Tree> children() {
    return Collections.emptyList();
  }
}
