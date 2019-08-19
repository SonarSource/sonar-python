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
package org.sonar.python.api.tree;

import com.sonar.sslr.api.AstNode;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.tree.PyFileInputTreeImpl;

public class PythonTreeMaker {

  public PyFileInputTree fileInput(AstNode astNode) {
    List<PyStatementTree> statements = astNode.getChildren(PythonGrammar.STATEMENT).stream().map(this::statement).collect(Collectors.toList());
    return new PyFileInputTreeImpl(astNode, statements);
  }

  private PyStatementTree statement(AstNode astNode) {
    if (astNode.is(PythonGrammar.IF_STMT)) {
    }
    return null;
  }


}
