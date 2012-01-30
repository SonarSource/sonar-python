/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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

package org.sonar.plugins.python;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;
import org.sonar.api.resources.Qualifiers;
import org.sonar.api.resources.Scopes;

public class PythonFileTest {
  @Test
  public void testStandaloneFile() throws Exception {
    String fname = "main.py";
    String relpath = "main.py";
    testThisCase("/tmp", fname, relpath,
                 null, fname, relpath);
  }

  @Test
  public void testFileInSubDir() throws Exception {
    String fname = "main.py";
    String relpath = "src/main.py";
    testThisCase("/tmp", fname, relpath,
                 null, fname, relpath);
  }

  @Test
  public void testFileInPackage() throws Exception {
    //test with an ad-hoc build python package
  }
  
  @Test
  public void testFileInSubPackage() throws Exception {
    //test with an ad-hoc build python package
  }
  
  private void testThisCase(String basedir,
                            String fname,
                            String relpath,
                            PythonPackage parent,
                            String name,
                            String longname) throws Exception {
    PythonFile pyfile = new PythonFile(relpath, new java.io.File(basedir, relpath));
    
    assertEquals (pyfile.getParent(), parent);
    assertEquals (pyfile.getName(), fname);
    assertEquals (pyfile.getLongName(), relpath);

    assertEquals (pyfile.getLanguage(), Python.INSTANCE);
    assertEquals (pyfile.getScope(), Scopes.FILE);
    assertEquals (pyfile.getQualifier(), Qualifiers.FILE);
  }
}
