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
import org.sonar.plugins.python.api.tree.DynamicObjectInfoStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class DynamicObjectInfoStatementImpl extends PyTree implements DynamicObjectInfoStatement {


  private final List<Token> questionMarksBefore;
  private final List<Tree> children;
  private final List<Token> questionMarksAfter;

  public DynamicObjectInfoStatementImpl(List<Token> questionMarksBefore, List<Tree> children, List<Token> questionMarksAfter) {
    this.questionMarksBefore = questionMarksBefore;
    this.children = children;
    this.questionMarksAfter = questionMarksAfter;
  }


  @Override
  public void accept(TreeVisitor visitor) {
    // no op
  }

  @Override
  public Kind getKind() {
    return Kind.DYNAMIC_OBJECT_INFO_STATEMENT;
  }

  @Override
  List<Tree> computeChildren() {
    var result = new ArrayList<Tree>();
    result.addAll(questionMarksBefore);
    result.addAll(children);
    result.addAll(questionMarksAfter);
    return result;
  }
}
