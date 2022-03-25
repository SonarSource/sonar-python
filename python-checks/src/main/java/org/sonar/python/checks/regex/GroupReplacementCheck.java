/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.checks.regex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.CapturingGroupTree;
import org.sonarsource.analyzer.commons.regex.ast.RegexBaseVisitor;

import static org.sonar.python.tree.TreeUtils.nthArgumentOrKeyword;


@Rule(key = "S6328")
public class GroupReplacementCheck extends AbstractRegexCheck {

  private static final String MESSAGE = "Referencing non-existing group%s: %s.";
  private static final Pattern REFERENCE_PATTERN = Pattern.compile("\\$(\\d+)|\\$\\{(\\d+)}|\\\\(\\d+)");

  @Override
  protected Map<String, Integer> lookedUpFunctions() {
    Map<String, Integer> result = new HashMap<>();
    result.put("re.sub", 4);
    return result;
  }

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    GroupFinder groupFinder = new GroupFinder();
    groupFinder.visit(regexParseResult);
    checkReplacement(regexFunctionCall, groupFinder.groups);
  }

  private void checkReplacement(CallExpression tree, Set<CapturingGroupTree> groups) {
    RegularArgument regArg = nthArgumentOrKeyword(1, "replacement", tree.arguments());
    if(regArg == null){
      return;
    }
    if(regArg.expression().is(Tree.Kind.STRING_LITERAL)){
      StringLiteral expression = (StringLiteral) regArg.expression();
      List<Integer> references = collectReferences(expression.trimmedQuotesValue());
      references.removeIf(reference -> groups.stream().anyMatch(group -> group.getGroupNumber() == reference));
      if (!references.isEmpty()) {
        List<String> stringReferences = references.stream().map(String::valueOf).collect(Collectors.toList());
        regexContext.addIssue(expression, String.format(MESSAGE, references.size() == 1 ? "" : "s", String.join(", ", stringReferences)));
      }
    }
  }

  private static List<Integer> collectReferences(String replacement) {
    Matcher match = REFERENCE_PATTERN.matcher(replacement);
    List<Integer> references = new ArrayList<>();
    while (match.find()) {
      // extract reference number out of one of the possible 3 groups of the regex
      for (int i = 1; i <= 3; i++) {
        Optional.ofNullable(match.group(i)).map(Integer::valueOf).filter(ref -> ref != 0).ifPresent(references::add);
      }
    }
    return references;
  }

  static class GroupFinder extends RegexBaseVisitor {

    private final Set<CapturingGroupTree> groups = new HashSet<>();

    @Override
    public void visitCapturingGroup(CapturingGroupTree group) {
      groups.add(group);
      super.visitCapturingGroup(group);
    }
  }
}

