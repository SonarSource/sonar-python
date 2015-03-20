/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.checks;

import org.junit.Test;
import org.sonar.python.PythonAstScanner;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.checks.CheckMessagesVerifier;

import java.io.File;

public class ModuleNameCheckTest {

  private ModuleNameCheck check = new ModuleNameCheck();

  @Test
  public void bad_name() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/badModule_name.py"), check);
    String format = "(([a-z_][a-z0-9_]*)|([A-Z][a-zA-Z0-9]+))$";
    String message = "Rename this module to match this regular expression: \"%s\".";
    CheckMessagesVerifier.verify(file.getCheckMessages())
        .next().withMessage(String.format(message, format))
        .noMore();
  }

  @Test
  public void good_name_camel_case() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/ModuleName.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())
        .noMore();
  }

  @Test
  public void good_name_snake_case() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/module_name.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())
        .noMore();
  }

}
