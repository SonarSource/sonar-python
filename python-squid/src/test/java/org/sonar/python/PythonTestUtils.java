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

import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.impl.Parser;
import java.nio.charset.StandardCharsets;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.tree.PythonTreeMaker;

public final class PythonTestUtils {

  private static final Parser<Grammar> p = PythonParser.create(new PythonConfiguration(StandardCharsets.UTF_8));
  private static final PythonTreeMaker pythonTreeMaker = new PythonTreeMaker();

  private PythonTestUtils() {
  }

  public static String appendNewLine(String s) {
    return s + "\n";
  }

  public static FileInput parse(String... lines) {
    String code = String.join(System.getProperty("line.separator"), lines);
    FileInput tree = pythonTreeMaker.fileInput(p.parse(code));
    new SymbolTableBuilder().visitFileInput(tree);
    return tree;
  }


}
