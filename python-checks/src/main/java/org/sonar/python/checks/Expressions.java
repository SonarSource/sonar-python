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
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyDictionaryLiteralTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyListLiteralTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyNumericLiteralTree;
import org.sonar.python.api.tree.PyStringLiteralTree;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.Symbol;
import org.sonar.python.semantic.Usage;

public class Expressions {

  private static final Set<String> ZERO_VALUES = new HashSet<>(Arrays.asList("0", "0.0", "0j"));

  private Expressions() {
  }

  // https://docs.python.org/3/library/stdtypes.html#truth-value-testing
  public static boolean isFalsy(@Nullable PyExpressionTree expression) {
    if (expression == null) {
      return false;
    }
    switch (expression.getKind()) {
      case NAME: 
        return "False".equals(((PyNameTree) expression).name());
      case NONE:
        return true;
      case STRING_LITERAL:
        return ((PyStringLiteralTree) expression).trimmedQuotesValue().isEmpty();
      case NUMERIC_LITERAL:
        return ZERO_VALUES.contains(((PyNumericLiteralTree) expression).valueAsString());
      case LIST_LITERAL:
        return ((PyListLiteralTree) expression).elements().expressions().isEmpty();
      case TUPLE:
        return ((PyTupleTree) expression).elements().isEmpty();
      case DICTIONARY_LITERAL:
        return ((PyDictionaryLiteralTree) expression).elements().isEmpty();
      default:
        return false;
    }
  }

  public static PyExpressionTree singleAssignedValue(PyNameTree name) {
    Symbol symbol = name.symbol();
    if (symbol == null) {
      return null;
    }
    PyExpressionTree result = null;
    for (Usage usage : symbol.usages()) {
      if (usage.kind() == Usage.Kind.ASSIGNMENT_LHS) {
        if (result != null) {
          return null;
        }
        Tree parent = usage.tree().parent();
        if (parent.is(Kind.EXPRESSION_LIST) && parent.parent().is(Kind.ASSIGNMENT_STMT)) {
          result = ((PyAssignmentStatementTree) parent.parent()).assignedValue();
        } else {
          return null;
        }
      } else if (usage.isBindingUsage()) {
        return null;
      }
    }
    return result;
  }
}
