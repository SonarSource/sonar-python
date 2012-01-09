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

import org.junit.Test;
import org.sonar.api.resources.Qualifiers;
import org.sonar.api.resources.Resource;

public class PythonPackageTest {

  @Test
  public void testPackage() throws Exception {
    String fname = "src/package/file";
    PythonPackage pypack = new PythonPackage(fname);

    // assert(pypack.getParent() != null);
    assert (pypack.getLanguage() == Python.INSTANCE);
    assert (pypack.getName() == fname);
    // assert(pypack.getLongName() == fname);
    assert (pypack.getScope() == Resource.SCOPE_SPACE);
    assert (pypack.getQualifier() == Qualifiers.PACKAGE);
  }
}
