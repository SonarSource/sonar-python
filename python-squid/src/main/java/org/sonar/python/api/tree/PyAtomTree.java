package org.sonar.python.api.tree;

public interface PyAtomTree extends PyExpressionTree {
  PyExpressionTree atom();
}
