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

import com.sonar.sslr.impl.Parser;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.tree.BaseTreeVisitor;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.checks.Expressions.isFalsy;

public class ExpressionsTest {

  private Parser parser = PythonParser.create(new PythonConfiguration(StandardCharsets.UTF_8));

  @Test
  public void falsy() {
    assertThat(isFalsy(null)).isFalse();

    assertThat(isFalsy(exp("True"))).isFalse();
    assertThat(isFalsy(exp("False"))).isTrue();
    assertThat(isFalsy(exp("x"))).isFalse();
    assertThat(isFalsy(exp("None"))).isTrue();

    assertThat(isFalsy(exp("''"))).isTrue();
    assertThat(isFalsy(exp("'x'"))).isFalse();
    assertThat(isFalsy(exp("' '"))).isFalse();
    assertThat(isFalsy(exp("''"))).isTrue();
    assertThat(isFalsy(exp("\"\""))).isTrue();
    assertThat(isFalsy(exp("'' 'x'"))).isFalse();
    assertThat(isFalsy(exp("'' ''"))).isTrue();

    assertThat(isFalsy(exp("1"))).isFalse();
    assertThat(isFalsy(exp("0"))).isTrue();
    assertThat(isFalsy(exp("0.0"))).isTrue();
    assertThat(isFalsy(exp("0j"))).isTrue();
    assertThat(isFalsy(exp("3.14"))).isFalse();

    assertThat(isFalsy(exp("[x]"))).isFalse();
    assertThat(isFalsy(exp("[]"))).isTrue();
    assertThat(isFalsy(exp("(x)"))).isFalse();
    assertThat(isFalsy(exp("()"))).isTrue();
    assertThat(isFalsy(exp("{x:y}"))).isFalse();
    assertThat(isFalsy(exp("{x}"))).isFalse();
    assertThat(isFalsy(exp("{}"))).isTrue();

    assertThat(isFalsy(exp("x.y"))).isFalse();
  }

  private Expression exp(String code) {
    return exp(parse(code));
  }

  private Expression exp(Tree tree) {
    if (tree instanceof Expression) {
      return (Expression) tree;
    }
    for (Tree child : tree.children()) {
      Expression exp = exp(child);
      if (exp != null) {
        return exp;
      }
    }
    return null;
  }

  private FileInput parse(String code) {
    return new PythonTreeMaker().fileInput(parser.parse(code));
  }

  @Test
  public void singleAssignedValue() {
    assertThat(lastNameValue("x = 42; x").getKind()).isEqualTo(Kind.NUMERIC_LITERAL);
    assertThat(lastNameValue("x = ''; x").getKind()).isEqualTo(Kind.STRING_LITERAL);
    assertThat(lastNameValue("(x, y) = (42, 43); x")).isNull();
    assertThat(lastNameValue("x = 42; import x; x")).isNull();
    assertThat(lastNameValue("x = 42; x = 43; x")).isNull();
    assertThat(lastNameValue("x = 42; y")).isNull();
  }

  private Expression lastNameValue(String code) {
    FileInput root = parse(code);
    new SymbolTableBuilder().visitFileInput(root);
    NameVisitor nameVisitor = new NameVisitor();
    root.accept(nameVisitor);
    List<Name> names = nameVisitor.names;
    return Expressions.singleAssignedValue(names.get(names.size() - 1));
  }

  private static class NameVisitor extends BaseTreeVisitor {
    private List<Name> names = new ArrayList<>();

    @Override
    public void visitName(Name pyNameTree) {
      names.add(pyNameTree);
    }
  }

}
