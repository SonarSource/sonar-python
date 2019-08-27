package org.sonar.python.tree;

import com.sonar.sslr.api.AstNode;
import org.sonar.python.api.tree.PyAtomTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;

public class PyAtomTreeImpl extends PyExpressionTreeImpl implements PyAtomTree {
  private final PyExpressionTree expression;

  public PyAtomTreeImpl(AstNode astNode, PyExpressionTree expression) {
    super(astNode);
    this.expression = expression;
  }

  @Override
  public PyExpressionTree atom() {
    return expression;
  }

  @Override
  public Kind getKind() {
    return Kind.ATOM;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitAtom(this);
  }
}
