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

import java.util.List;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.v2.PythonType;

public class SetLiteralImpl extends DictOrSetLiteralImpl<Expression> implements SetLiteral {

  private PythonType pythonType = PythonType.UNKNOWN;

  public SetLiteralImpl(Token lCurlyBrace, List<Expression> elements, List<Token> commas, Token rCurlyBrace) {
    super(lCurlyBrace, commas, elements, rCurlyBrace);
  }
  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitSetLiteral(this);
  }

  @Override
  public Kind getKind() {
    return Kind.SET_LITERAL;
  }

  @Override
  public InferredType type() {
    return InferredTypes.SET;
  }

  @Override
  public PythonType typeV2() {
    return pythonType;
  }

  public SetLiteralImpl typeV2(PythonType pythonType) {
    this.pythonType = pythonType;
    return this;
  }
}
