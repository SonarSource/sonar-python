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
package org.sonar.python;

import com.intellij.psi.PsiElement;
import com.intellij.psi.tree.IElementType;
import com.jetbrains.python.psi.PyFile;
import com.jetbrains.python.psi.PyRecursiveElementVisitor;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import org.sonar.python.PythonCheck.PreciseIssue;

public class SubscriptionVisitor extends PyRecursiveElementVisitor {

  private final Map<IElementType, List<ConsumerWrapper>> consumers = new HashMap<>();
  private final PythonVisitorContext pythonVisitorContext;
  private PsiElement currentElement;

  public static void analyze(Collection<PythonCheck> checks, PythonVisitorContext pythonVisitorContext, PyFile pyFile) {
    SubscriptionVisitor subscriptionVisitor = new SubscriptionVisitor(checks, pythonVisitorContext);
    pyFile.accept(subscriptionVisitor);
  }

  private SubscriptionVisitor(Collection<PythonCheck> checks, PythonVisitorContext pythonVisitorContext) {
    this.pythonVisitorContext = pythonVisitorContext;
    for (PythonCheck check : checks) {
      check.initialize((elementType, consumer) -> {
        List<ConsumerWrapper> elementConsumers = consumers.computeIfAbsent(elementType, c -> new ArrayList<>());
        elementConsumers.add(new ConsumerWrapper(check, consumer));
      });
    }
  }

  @Override
  public void visitElement(PsiElement element) {
    currentElement = element;
    List<ConsumerWrapper> elementConsumers = consumers.get(element.getNode().getElementType());
    if (elementConsumers != null) {
      for (ConsumerWrapper consumer : elementConsumers) {
        consumer.execute();
      }
    }
    super.visitElement(element);
  }

  private class ConsumerWrapper implements SubscriptionContext {
    private final PythonCheck check;
    private final Consumer<SubscriptionContext> consumer;

    ConsumerWrapper(PythonCheck check, Consumer<SubscriptionContext> consumer) {
      this.check = check;
      this.consumer = consumer;
    }

    public void execute() {
      consumer.accept(this);
    }

    @Override
    public PsiElement syntaxNode() {
      return SubscriptionVisitor.this.currentElement;
    }

    @Override
    public PreciseIssue addIssue(PsiElement element, @Nullable String message) {
      return pythonVisitorContext.addIssue(check, element, message);
    }

    @Override
    public PreciseIssue addIssue(IssueLocation issueLocation) {
      return pythonVisitorContext.addIssue(check, issueLocation);
    }

    @Override
    public PreciseIssue addFileIssue(@Nullable String message) {
      return pythonVisitorContext.addIssue(check, IssueLocation.atFileLevel(message));
    }
  }
}
