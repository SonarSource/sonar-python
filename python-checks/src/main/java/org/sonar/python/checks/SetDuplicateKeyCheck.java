/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S5781")
public class SetDuplicateKeyCheck extends AbstractDuplicateKeyCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.SET_LITERAL, ctx -> {
      SetLiteral setLiteral = (SetLiteral) ctx.syntaxNode();
      Set<Integer> issueIndexes = new HashSet<>();
      if (setLiteral.elements().size() > SIZE_THRESHOLD) {
        return;
      }
      for (int i = 0; i < setLiteral.elements().size(); i++) {
        if (issueIndexes.contains(i)) {
          continue;
        }
        Expression key = setLiteral.elements().get(i);
        List<Tree> duplicateKeys = findIdenticalKeys(i, setLiteral.elements(), issueIndexes);
        if (!duplicateKeys.isEmpty()) {
          PreciseIssue issue = ctx.addIssue(key, "Change or remove duplicates of this key.");
          duplicateKeys.forEach(d -> issue.secondary(d, "Duplicate key"));
        }
      }
    });
  }

  private List<Tree> findIdenticalKeys(int startIndex, List<Expression> elements, Set<Integer> issueIndexes) {
    Expression key = elements.get(startIndex);
    List<Tree> duplicates = new ArrayList<>();
    for (int i = startIndex + 1; i < elements.size(); i++) {
      Expression comparedKey = elements.get(i);
      if (isSameKey(key, comparedKey)) {
        issueIndexes.add(i);
        duplicates.add(comparedKey);
      }
    }
    return duplicates;
  }
}
