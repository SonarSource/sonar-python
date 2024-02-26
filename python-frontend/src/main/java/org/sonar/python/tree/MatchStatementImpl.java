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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.sonar.plugins.python.api.tree.CaseBlock;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.MatchStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class MatchStatementImpl extends PyTree implements MatchStatement {
  private final Token matchKeyword;
  private final Expression subjectExpression;
  private final Token colon;
  private final Token newLine;
  private final Token indent;
  private final List<CaseBlock> caseBlocks;
  private final Token dedent;

  public MatchStatementImpl(Token matchKeyword, Expression subjectExpression, Token colon, Token newLine, Token indent,
    List<CaseBlock> caseBlocks, Token dedent) {

    this.matchKeyword = matchKeyword;
    this.subjectExpression = subjectExpression;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.caseBlocks = caseBlocks;
    this.dedent = dedent;
  }


  @Override
  public Token matchKeyword() {
    return matchKeyword;
  }

  @Override
  public Expression subjectExpression() {
    return subjectExpression;
  }

  @Override
  public Token colon() {
    return colon;
  }

  @Override
  public List<CaseBlock> caseBlocks() {
    return caseBlocks;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitMatchStatement(this);
  }

  @Override
  public Kind getKind() {
    return Kind.MATCH_STMT;
  }

  @Override
  List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>(Arrays.asList(matchKeyword, subjectExpression, colon, newLine, indent));
    children.addAll(caseBlocks);
    children.add(dedent);
    return children;
  }
}
