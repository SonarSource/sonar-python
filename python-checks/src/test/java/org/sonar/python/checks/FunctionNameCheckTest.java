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

public class FunctionNameCheckTest {

  @Test
  public void test() throws Exception {
    FunctionNameCheck check = new FunctionNameCheck();
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/functionName.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())
      .next().atLine(4).withMessage(
        "Rename function \"Badly_Named_Function\" to match the regular expression ^[a-z_][a-z0-9_]{2,30}$.")
      .next().atLine(10).withMessage(
        "Rename function \"too_long_function_name_because_it_has_more_than_30_characters\" "
          + "to match the regular expression ^[a-z_][a-z0-9_]{2,30}$.")
      .noMore();
  }

}
