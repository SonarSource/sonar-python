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
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S5780")
public class DictionaryDuplicateKeyCheck extends AbstractDuplicateKeyCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DICTIONARY_LITERAL, ctx -> {
      DictionaryLiteral dictionaryLiteral = (DictionaryLiteral) ctx.syntaxNode();
      Set<Integer> issueIndexes = new HashSet<>();
      if (dictionaryLiteral.elements().size() > SIZE_THRESHOLD) {
        return;
      }
      for (int i = 0; i < dictionaryLiteral.elements().size(); i++) {
        if (!dictionaryLiteral.elements().get(i).is(Tree.Kind.KEY_VALUE_PAIR) || issueIndexes.contains(i)) {
          continue;
        }
        Expression key = ((KeyValuePair) dictionaryLiteral.elements().get(i)).key();
        List<Tree> duplicateKeys = findIdenticalKeys(i, dictionaryLiteral.elements(), issueIndexes);
        if (!duplicateKeys.isEmpty()) {
          PreciseIssue issue = ctx.addIssue(key, "Change or remove duplicates of this key.");
          duplicateKeys.forEach(d -> issue.secondary(d, "Duplicate key"));
        }
      }
    });
  }

  private List<Tree> findIdenticalKeys(int startIndex, List<DictionaryLiteralElement> elements, Set<Integer> issueIndexes) {
    Expression key = ((KeyValuePair) elements.get(startIndex)).key();
    List<Tree> duplicates = new ArrayList<>();
    for (int i = startIndex + 1; i < elements.size(); i++) {
      if (!elements.get(i).is(Tree.Kind.KEY_VALUE_PAIR)) {
        continue;
      }
      Expression comparedKey = ((KeyValuePair) elements.get(i)).key();
      if (isSameKey(key, comparedKey)) {
        issueIndexes.add(i);
        duplicates.add(comparedKey);
      }
    }
    return duplicates;
  }
}
