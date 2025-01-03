/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.plugins.python;

import java.util.Objects;
import org.sonar.api.config.Configuration;
import org.sonar.api.resources.AbstractLanguage;

import static org.sonar.plugins.python.Python.filterEmptyStrings;

public class IPynb extends AbstractLanguage {

  public static final String KEY = "ipynb";

  private static final String[] DEFAULT_FILE_SUFFIXES = {KEY};
  private final Configuration configuration;

  public IPynb(Configuration configuration) {
    super(KEY, "IPython Notebooks");
    this.configuration = configuration;
  }

  @Override
  public String[] getFileSuffixes() {
    String[] suffixes = filterEmptyStrings(configuration.getStringArray(PythonPlugin.IPYNB_FILE_SUFFIXES_KEY));
    return suffixes.length == 0 ? DEFAULT_FILE_SUFFIXES : suffixes;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()){
      return false;
    }
    if (!super.equals(o)) {
      return false;
    }
    IPynb iPynb = (IPynb) o;
    return Objects.equals(configuration, iPynb.configuration);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), configuration);
  }
}
