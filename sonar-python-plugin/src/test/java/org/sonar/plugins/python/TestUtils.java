/*
 * Sonar Python Plugin
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
package org.sonar.plugins.python;

import static org.junit.Assert.fail;
import static org.mockito.Matchers.anyObject;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.File;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.List;
import java.util.LinkedList;

import org.apache.commons.configuration.Configuration;
import org.sonar.api.resources.InputFile;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.ProjectFileSystem;
import org.sonar.api.resources.Language;
import org.sonar.api.rules.Rule;
import org.sonar.api.rules.RuleFinder;
import org.sonar.api.rules.RuleQuery;

public class TestUtils{
  public static File loadResource(String resourceName) {
    URL resource = TestUtils.class.getResource(resourceName);
    File resourceAsFile = null;
    try{
      resourceAsFile = new File(resource.toURI());
    } catch (URISyntaxException e) {
      System.out.println("Cannot load resource: " + resourceName);
    }
    
    return resourceAsFile;
  }
  
  /**
   * @return default mock project
   */
  public static Project mockProject() {
    return mockProject(loadResource("/org/sonar/plugins/python/"));
  }
  
  /**
   * Mock project
   * @param baseDir project base dir
   * @return mocked project
   */
  public static Project mockProject(File baseDir) {
    List<InputFile> mainFiles = new LinkedList<InputFile>();
    List<InputFile> testFiles = new LinkedList<InputFile>();
    
    ProjectFileSystem fileSystem = mock(ProjectFileSystem.class);
    when(fileSystem.getBasedir()).thenReturn(baseDir);
    when(fileSystem.getSourceCharset()).thenReturn(Charset.defaultCharset());
    when(fileSystem.mainFiles(Python.KEY)).thenReturn(mainFiles);
    when(fileSystem.testFiles(Python.KEY)).thenReturn(testFiles);

    Project project = mock(Project.class);
    when(project.getFileSystem()).thenReturn(fileSystem);
    Language lang = mockLanguage();
    when(project.getLanguage()).thenReturn(lang);
    
    return project;
  }
  
  public static Python mockLanguage(){
    Python lang = mock(Python.class);
    return lang;
  }
}
