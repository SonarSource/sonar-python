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
package org.sonar.python.checks.hotspots;

import com.intellij.psi.PsiElement;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyImportElement;
import com.jetbrains.python.psi.PyReferenceExpression;
import com.jetbrains.python.psi.PyTargetExpression;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.checks.AbstractCallExpressionBase;

@Rule(key = "S4823")
public class CommandLineArgsCheck extends AbstractCallExpressionBase {
  private static final String MESSAGE = "Make sure that command line arguments are used safely here.";
  private static final Set<String> questionableFunctions = immutableSet("argparse.ArgumentParser.__init__", "optparse.OptionParser.__init__");

  @Override
  public void initialize(Context context) {
    super.initialize(context);
    context.registerSyntaxNodeConsumer(PyElementTypes.REFERENCE_EXPRESSION, ctx -> {
      PyReferenceExpression node = (PyReferenceExpression) ctx.syntaxNode();
      if (node.getParent() instanceof PyImportElement) {
        return;
      }
      PsiElement resolve = node.getReference().resolve();
      if (resolve instanceof PyTargetExpression && "sys.argv".equals(((PyTargetExpression) resolve).getQualifiedName())) {
        ctx.addIssue(node, message());
      }
    });
  }

  @Override
  protected Set<String> functionsToCheck() {
    return questionableFunctions;
  }

  @Override
  protected String message() {
    return MESSAGE;
  }
}
