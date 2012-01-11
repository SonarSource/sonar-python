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

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.Extension;
import org.sonar.api.Plugin;
import org.sonar.api.Properties;
import org.sonar.api.Property;

@Properties(
  @Property(key = PythonPlugin.PYLINT_CONFIG_KEY,
	    defaultValue = "",
	    name = "pylint configuration",
	    description = "Path to the pylint configuration file to use in pylint analysis",
	    global = true,
	    project = true)
)
public class PythonPlugin implements Plugin {
  private static final String PROPERTY_PREFIX = "sonar.python.";
  protected static final String PYLINT_CONFIG_KEY = PROPERTY_PREFIX + "pylint_config";
  protected static final Logger LOG = LoggerFactory.getLogger(PythonPlugin.class);

  public String getKey() {
    return "Python Plugin";
  }

  public String getName() {
    return "Python";
  }

  public String getDescription() {
    return "Analysis of Python projects";
  }

  public List<Class<? extends Extension>> getExtensions() {
    List<Class<? extends Extension>> list = new ArrayList<Class<? extends Extension>>();

    list.add(Python.class);
    list.add(PythonSourceImporter.class);
    list.add(PythonSquidSensor.class);
    list.add(PythonComplexitySensor.class);
    list.add(PythonViolationsSensor.class);
    list.add(PythonRuleRepository.class);
    list.add(PythonDefaultProfile.class);
    list.add(PythonColorizer.class);

    return list;
  }
}
