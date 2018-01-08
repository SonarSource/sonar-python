/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import com.sonar.sslr.api.Token;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonTokenType;

/**
 * Extractor of docstring tokens.
 * <p>
 * Reminder: a docstring is a string literal that occurs as the first statement
 * in a module, function, class, or method definition.
 */
public class DocstringExtractor {

  public static final Set<AstNodeType> DOCUMENTABLE_NODE_TYPES = initializeDocumentableNodeTypes();

  private DocstringExtractor() {
  }

  private static Set<AstNodeType> initializeDocumentableNodeTypes() {
    Set<AstNodeType> set = new HashSet<>();
    set.add(PythonGrammar.FILE_INPUT);
    set.add(PythonGrammar.FUNCDEF);
    set.add(PythonGrammar.CLASSDEF);
    return Collections.unmodifiableSet(set);
  }

  public static Token extractDocstring(AstNode documentableNode) {
    if (documentableNode.is(PythonGrammar.FILE_INPUT)) {
      return extractModuleDocstring(documentableNode);
    }
    return extractDocstringFromFirstSuite(documentableNode);
  }

  private static Token extractModuleDocstring(AstNode astNode) {
    AstNode firstStatement = astNode.getFirstChild(PythonGrammar.STATEMENT);
    AstNode firstSimpleStmt = null;
    if (firstStatement != null) {
      firstSimpleStmt = getFirstSimpleStmt(firstStatement);
    }
    return extractDocstringFromSimpleStmt(firstSimpleStmt);
  }

  private static AstNode getFirstSimpleStmt(AstNode statement) {
    AstNode stmtList = statement.getFirstChild(PythonGrammar.STMT_LIST);
    if (stmtList != null) {
      return stmtList.getFirstChild(PythonGrammar.SIMPLE_STMT);
    }
    return null;
  }

  private static Token extractDocstringFromFirstSuite(AstNode documentableNode) {
    AstNode suite = documentableNode.getFirstChild(PythonGrammar.SUITE);
    AstNode firstStatement = suite.getFirstChild(PythonGrammar.STATEMENT);
    AstNode firstSimpleStmt;
    if (firstStatement == null) {
      firstSimpleStmt = suite.getFirstChild(PythonGrammar.STMT_LIST).getFirstChild(PythonGrammar.SIMPLE_STMT);
    } else {
      firstSimpleStmt = getFirstSimpleStmt(firstStatement);
    }
    return extractDocstringFromSimpleStmt(firstSimpleStmt);
  }

  private static Token extractDocstringFromSimpleStmt(@Nullable AstNode firstSimpleStmt) {
    if (firstSimpleStmt != null) {
      Token token = firstSimpleStmt.getToken();
      if (token.getType().equals(PythonTokenType.STRING)) {
        return token;
      }
    }
    return null;
  }

}
