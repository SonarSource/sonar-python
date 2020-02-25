/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import javax.annotation.CheckForNull;
import org.assertj.core.api.Assertions;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.tree.PythonTreeMaker;

public final class PythonTestUtils {

  private static final PythonParser p = PythonParser.create();
  private static final PythonTreeMaker pythonTreeMaker = new PythonTreeMaker();

  private PythonTestUtils() {
  }

  public static String appendNewLine(String s) {
    return s + "\n";
  }

  public static FileInput parse(String... lines) {
    return parse(new SymbolTableBuilder(pythonFile("")), lines);
  }

  public static FileInput parse(SymbolTableBuilder symbolTableBuilder, String... lines) {
    String code = String.join(System.getProperty("line.separator"), lines);
    FileInput tree = pythonTreeMaker.fileInput(p.parse(code));
    symbolTableBuilder.visitFileInput(tree);
    return tree;
  }

  public static FileInput parseWithoutSymbols(String... lines) {
    String code = String.join(System.getProperty("line.separator"), lines);
    return pythonTreeMaker.fileInput(p.parse(code));
  }


  @CheckForNull
  public static <T extends Tree> T getFirstChild(Tree tree, Predicate<Tree> predicate) {
    for (Tree child : tree.children()) {
      if (predicate.test(child)) {
        return (T) child;
      }
      Tree firstChild = getFirstChild(child, predicate);
      if (firstChild != null) {
        return (T) firstChild;
      }
    }
    return null;
  }

  public static <T extends Tree> List<T> getAllDescendant(Tree tree, Predicate<Tree> predicate) {
    List<T> res = new ArrayList<>();
    for (Tree child : tree.children()) {
      if (predicate.test(child)) {
        res.add((T) child);
      }
      res.addAll(getAllDescendant(child, predicate));
    }
    return res;
  }

  public static <T extends Tree> T getUniqueDescendant(Tree tree, Predicate<Tree> predicate) {
    List<T> descendants = getAllDescendant(tree, predicate);
    Assertions.assertThat(descendants).hasSize(1);
    return descendants.get(0);
  }

  public static <T extends Tree> T getLastDescendant(Tree tree, Predicate<Tree> predicate) {
    List<T> descendants = getAllDescendant(tree, predicate);
    Assertions.assertThat(descendants).isNotEmpty();
    return descendants.get(descendants.size() - 1);
  }

  public static PythonFile pythonFile(String fileName) {
    PythonFile pythonFile = Mockito.mock(PythonFile.class);
    Mockito.when(pythonFile.fileName()).thenReturn(fileName);
    try {
      Mockito.when(pythonFile.uri()).thenReturn(Files.createTempFile(fileName, "py").toUri());
    } catch (IOException e) {
      throw new IllegalStateException("Cannot create temporary file");
    }
    return pythonFile;
  }
}
