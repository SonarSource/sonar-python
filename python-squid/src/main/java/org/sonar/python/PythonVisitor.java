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
import java.util.List;
import java.util.Set;

public class PythonVisitor {

  private PythonVisitorContext context;

  public Set<AstNodeType> subscribedKinds() {
    return Collections.emptySet();
  }

  public void visitFile(AstNode node) {
    // default implementation does nothing
  }

  public void leaveFile(AstNode node) {
    // default implementation does nothing
  }

  public void visitNode(AstNode node) {
    // default implementation does nothing
  }

  public void visitToken(Token token) {
    // default implementation does nothing
  }

  public void leaveNode(AstNode node) {
    // default implementation does nothing
  }

  public PythonVisitorContext getContext() {
    return context;
  }

  public void scanFile(PythonVisitorContext context) {
    this.context = context;
    AstNode tree = context.rootTree();
    if (tree != null) {
      visitFile(tree);
      scanNode(tree, subscribedKinds());
      leaveFile(tree);
    }
  }

  public void scanNode(AstNode node) {
    scanNode(node, subscribedKinds());
  }

  private void scanNode(AstNode node, Set<AstNodeType> subscribedKinds) {
    boolean isSubscribedType = subscribedKinds.contains(node.getType());

    if (isSubscribedType) {
      visitNode(node);
    }

    List<AstNode> children = node.getChildren();
    if (children.isEmpty()) {
      for (Token token : node.getTokens()) {
        visitToken(token);
      }
    } else {
      for (AstNode child : children) {
        scanNode(child, subscribedKinds);
      }
    }

    if (isSubscribedType) {
      leaveNode(node);
    }
  }

}
