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
package org.sonar.python.frontend;

import com.intellij.core.CoreASTFactory;
import com.intellij.core.CoreFileTypeRegistry;
import com.intellij.lang.Language;
import com.intellij.lang.LanguageASTFactory;
import com.intellij.lang.LanguageParserDefinitions;
import com.intellij.lang.MetaLanguage;
import com.intellij.lang.PsiBuilderFactory;
import com.intellij.lang.impl.PsiBuilderFactoryImpl;
import com.intellij.mock.MockApplication;
import com.intellij.mock.MockFileDocumentManagerImpl;
import com.intellij.mock.MockProject;
import com.intellij.openapi.Disposable;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.editor.impl.DocumentImpl;
import com.intellij.openapi.extensions.ExtensionPoint;
import com.intellij.openapi.extensions.ExtensionPointName;
import com.intellij.openapi.extensions.Extensions;
import com.intellij.openapi.extensions.ExtensionsArea;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.openapi.fileTypes.FileTypeRegistry;
import com.intellij.openapi.progress.ProgressManager;
import com.intellij.openapi.progress.impl.ProgressManagerImpl;
import com.intellij.openapi.util.Disposer;
import com.intellij.openapi.util.StaticGetter;
import com.intellij.psi.PsiFileFactory;
import com.intellij.psi.PsiManager;
import com.intellij.psi.impl.PsiFileFactoryImpl;
import com.intellij.psi.impl.PsiManagerImpl;
import com.jetbrains.python.PythonDialectsTokenSetContributor;
import com.jetbrains.python.PythonFileType;
import com.jetbrains.python.PythonLanguage;
import com.jetbrains.python.PythonParserDefinition;
import com.jetbrains.python.PythonTokenSetContributor;
import com.jetbrains.python.psi.PyFile;
import java.lang.reflect.Modifier;
import org.jetbrains.annotations.NotNull;

public class PythonParser {

  private static final Disposable TEST_DISPOSABLE = new TestDisposable();

  private static final PsiFileFactory psiFileFactory = psiFileFactory();

  public PyFile parse(String content) {
    return (PyFile) psiFileFactory.createFileFromText("test.py", PythonFileType.INSTANCE, content, System.currentTimeMillis(), false, false);
  }

  private static PsiFileFactory psiFileFactory() {
    CoreFileTypeRegistry fileTypeRegistry = new CoreFileTypeRegistry();
    fileTypeRegistry.registerFileType(PythonFileType.INSTANCE, "py");
    FileTypeRegistry.ourInstanceGetter = new StaticGetter<>(fileTypeRegistry);

    Disposable disposable = Disposer.newDisposable();

    MockApplication application = new MockApplication(disposable);
    FileDocumentManager fileDocMgr = new MockFileDocumentManagerImpl(DocumentImpl::new, null);
    application.registerService(FileDocumentManager.class, fileDocMgr);
    PsiBuilderFactoryImpl psiBuilderFactory = new PsiBuilderFactoryImpl();
    application.registerService(PsiBuilderFactory.class, psiBuilderFactory);
    application.registerService(ProgressManager.class, ProgressManagerImpl.class);
    ApplicationManager.setApplication(application, FileTypeRegistry.ourInstanceGetter, disposable);

    Extensions.getArea(null).registerExtensionPoint(MetaLanguage.EP_NAME.getName(), MetaLanguage.class.getName(), ExtensionPoint.Kind.INTERFACE);
    Extensions.registerAreaClass("IDEA_PROJECT", null);
    registerExtensionPoint(PythonDialectsTokenSetContributor.EP_NAME, PythonDialectsTokenSetContributor.class);
    registerExtension(PythonDialectsTokenSetContributor.EP_NAME, new PythonTokenSetContributor());

    MockProject project = new MockProject(null, disposable);

    LanguageParserDefinitions.INSTANCE.addExplicitExtension(PythonLanguage.getInstance(), new PythonParserDefinition());
    CoreASTFactory astFactory = new CoreASTFactory();
    LanguageASTFactory.INSTANCE.addExplicitExtension(PythonLanguage.getInstance(), astFactory);
    LanguageASTFactory.INSTANCE.addExplicitExtension(Language.ANY, astFactory);

    // https://github.com/JetBrains/intellij-community/blob/93b632941e406178dd5c78fe4d8fdf7d8c357355/platform/testFramework/src/com/intellij/testFramework/ParsingTestCase.java
    //new MockPsiManager()
    PsiManager psiManager = new PsiManagerImpl(project, fileDocMgr, psiBuilderFactory, null, null, null);
    return new PsiFileFactoryImpl(psiManager);
  }

  protected static <T> void registerExtension(@NotNull ExtensionPointName<T> extensionPointName, @NotNull T t) {
    registerExtension(Extensions.getRootArea(), extensionPointName, t);
  }

  public static <T> void registerExtension(@NotNull ExtensionsArea area, @NotNull ExtensionPointName<T> name, @NotNull T t) {
    registerExtensionPoint(area, name, (Class<T>)t.getClass());
    area.<T>getExtensionPoint(name.getName()).registerExtension(t, TEST_DISPOSABLE);
  }

  private static <T> void registerExtensionPoint(@NotNull ExtensionPointName<T> extensionPointName, @NotNull Class<T> aClass) {
    registerExtensionPoint(Extensions.getRootArea(), extensionPointName, aClass);
  }

  private static <T> void registerExtensionPoint(
    @NotNull ExtensionsArea area,
    @NotNull ExtensionPointName<T> extensionPointName,
    @NotNull Class<? extends T> aClass
  ) {
    if (!area.hasExtensionPoint(extensionPointName)) {
      ExtensionPoint.Kind kind = aClass.isInterface() || (aClass.getModifiers() & Modifier.ABSTRACT) != 0 ? ExtensionPoint.Kind.INTERFACE : ExtensionPoint.Kind.BEAN_CLASS;
      area.registerExtensionPoint(extensionPointName, aClass.getName(), kind, TEST_DISPOSABLE);
    }
  }

  protected static class TestDisposable implements Disposable {
    @Override
    public void dispose() {
    }
  }

}
