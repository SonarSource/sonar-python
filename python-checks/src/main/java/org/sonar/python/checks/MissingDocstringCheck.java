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
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import com.sonar.sslr.api.Token;
import java.util.Set;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.python.DocstringExtractor;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;

@Rule(key = MissingDocstringCheck.CHECK_KEY)
public class MissingDocstringCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1720";

  private static final Pattern EMPTY_STRING_REGEXP = Pattern.compile("([bruBRU]+)?('\\s*')|(\"\\s*\")|('''\\s*''')|(\"\"\"\\s*\"\"\")");
  private static final String MESSAGE_NO_DOCSTRING = "Add a docstring to this %s.";
  private static final String MESSAGE_EMPTY_DOCSTRING = "The docstring for this %s should not be empty.";

  private enum DeclarationType {
    MODULE("module"),
    CLASS("class"),
    METHOD("method"),
    FUNCTION("function");

    private final String value;

    DeclarationType(String value) {
      this.value = value;
    }
  }

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return DocstringExtractor.DOCUMENTABLE_NODE_TYPES;
  }

  @Override
  public void visitNode(AstNode astNode) {
    DeclarationType type = getType(astNode);
    Token docstring = DocstringExtractor.extractDocstring(astNode);
    if (docstring == null) {
      raiseIssueNoDocstring(astNode, type);
    } else if (EMPTY_STRING_REGEXP.matcher(docstring.getValue()).matches()) {
      raiseIssue(astNode, MESSAGE_EMPTY_DOCSTRING, type);
    }
  }

  private static DeclarationType getType(AstNode node) {
    if (node.is(PythonGrammar.FILE_INPUT)) {
      return DeclarationType.MODULE;
    } else if (node.is(PythonGrammar.FUNCDEF)) {
      if (CheckUtils.isMethodDefinition(node)) {
        return DeclarationType.METHOD;
      } else {
        return DeclarationType.FUNCTION;
      }
    } else if (node.is(PythonGrammar.CLASSDEF)) {
      return DeclarationType.CLASS;
    }
    return null;
  }

  private void raiseIssueNoDocstring(AstNode astNode, DeclarationType type) {
    if (type != DeclarationType.METHOD) {
      raiseIssue(astNode, MESSAGE_NO_DOCSTRING, type);
    }
  }

  private void raiseIssue(AstNode astNode, String message, DeclarationType type) {
    String finalMessage = String.format(message, type.value);
    if (type != DeclarationType.MODULE) {
      addIssue(getNameNode(astNode), finalMessage);
    } else {
      addFileIssue(finalMessage);
    }
  }

  private static AstNode getNameNode(AstNode astNode) {
    return astNode.getFirstChild(PythonGrammar.FUNCNAME, PythonGrammar.CLASSNAME);
  }

}
