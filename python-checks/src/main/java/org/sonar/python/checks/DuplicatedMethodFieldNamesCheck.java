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

import com.intellij.psi.PsiElement;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyClass;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.frontend.PythonTokenLocation;

@Rule(key = "S1845")
public class DuplicatedMethodFieldNamesCheck extends PythonCheck {
  private static final String MESSAGE = "Rename %s \"%s\" to prevent any misunderstanding/clash with %s \"%s\" defined on line %s";

  private static class TokenWithTypeInfo {
    private final PsiElement node;
    private final PythonTokenLocation token;
    private final String type;
    private final String name;

    TokenWithTypeInfo(PsiElement node, String name, String type) {
      this.node = node;
      this.type = type;
      this.name = name;
      this.token = new PythonTokenLocation(node);
    }

    String getValue() {
      return name;
    }

    int getLine() {
      return token.startLine();
    }

    String getType() {
      return type;
    }
  }

  private static class LineComparator implements Comparator<TokenWithTypeInfo>, Serializable {

    private static final long serialVersionUID = 4759444000993633906L;

    @Override
    public int compare(TokenWithTypeInfo t1, TokenWithTypeInfo t2) {
      return Integer.compare(t1.getLine(), t2.getLine());
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.CLASS_DECLARATION, ctx -> {
      PyClass classDecl = (PyClass) ctx.syntaxNode();
      List<TokenWithTypeInfo> allTokensWithInfo = classDecl.getInstanceAttributes().stream()
        .filter(attribute -> attribute.getName() != null)
        .map(attribute -> new TokenWithTypeInfo((PsiElement) attribute.getNode().getLastChildNode(), attribute.getName(), "field"))
        .collect(Collectors.toList());

      allTokensWithInfo.addAll(Arrays.stream(classDecl.getMethods())
        .filter(method -> method.getNameIdentifier() != null && method.getName() != null)
        .map(method -> new TokenWithTypeInfo(method.getNameIdentifier(), method.getName(), "method"))
        .collect(Collectors.toList()));

      allTokensWithInfo.addAll(classDecl.getClassAttributes().stream()
        .filter(attribute -> attribute.getName() != null)
        .map(attribute -> new TokenWithTypeInfo(attribute, attribute.getName(), "field"))
        .collect(Collectors.toList())
      );

      lookForDuplications(allTokensWithInfo, ctx);
    });
  }

  private void lookForDuplications(List<TokenWithTypeInfo> allTokensWithInfo, SubscriptionContext ctx) {
    allTokensWithInfo.sort(new LineComparator());
    for (int i = 1; i < allTokensWithInfo.size(); i++) {
      for (int j = i - 1; j >= 0; j--) {
        TokenWithTypeInfo token1 = allTokensWithInfo.get(j);
        TokenWithTypeInfo token2 = allTokensWithInfo.get(i);
        if (differOnlyByCapitalization(token1.getValue(), token2.getValue())) {
          ctx.addIssue(token2.node, getMessage(token1, token2))
            .secondary(token1.node, "Original");
          break;
        }
      }
    }
  }

  private static boolean differOnlyByCapitalization(String name1, String name2) {
    return name1.equalsIgnoreCase(name2) && !name1.equals(name2);
  }

  private static String getMessage(TokenWithTypeInfo token1, TokenWithTypeInfo token2) {
    return String.format(MESSAGE, token2.getType(), token2.getValue(), token1.getType(), token1.getValue(), token1.getLine());
  }

}
