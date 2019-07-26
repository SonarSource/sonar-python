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

import com.intellij.configurationStore.StreamProvider;
import com.intellij.core.CoreASTFactory;
import com.intellij.core.CoreEncodingProjectManager;
import com.intellij.core.CoreFileTypeRegistry;
import com.intellij.core.CoreModuleManager;
import com.intellij.core.CoreProjectJdkTable;
import com.intellij.core.CoreProjectScopeBuilder;
import com.intellij.execution.configurations.ConfigurationType;
import com.intellij.ide.util.AppPropertiesComponentImpl;
import com.intellij.ide.util.PropertiesComponent;
import com.intellij.ide.util.PropertiesComponentImpl;
import com.intellij.lang.Language;
import com.intellij.lang.LanguageASTFactory;
import com.intellij.lang.LanguageParserDefinitions;
import com.intellij.lang.MetaLanguage;
import com.intellij.lang.PsiBuilderFactory;
import com.intellij.lang.impl.PsiBuilderFactoryImpl;
import com.intellij.mock.MockApplication;
import com.intellij.mock.MockApplicationEx;
import com.intellij.mock.MockFileDocumentManagerImpl;
import com.intellij.mock.MockFileIndexFacade;
import com.intellij.mock.MockProject;
import com.intellij.openapi.Disposable;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.application.ex.ApplicationEx;
import com.intellij.openapi.components.RoamingType;
import com.intellij.openapi.editor.impl.DocumentImpl;
import com.intellij.openapi.editor.impl.EditorFactoryImpl;
import com.intellij.openapi.extensions.ExtensionPoint;
import com.intellij.openapi.extensions.ExtensionPointName;
import com.intellij.openapi.extensions.Extensions;
import com.intellij.openapi.extensions.ExtensionsArea;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.openapi.fileTypes.FileTypeConsumer;
import com.intellij.openapi.fileTypes.FileTypeFactory;
import com.intellij.openapi.fileTypes.FileTypeManager;
import com.intellij.openapi.fileTypes.FileTypeRegistry;
import com.intellij.openapi.fileTypes.impl.FileTypeManagerImpl;
import com.intellij.openapi.module.ModuleManager;
import com.intellij.openapi.options.EmptySchemesManager;
import com.intellij.openapi.options.SchemeManager;
import com.intellij.openapi.options.SchemeManagerFactory;
import com.intellij.openapi.options.SchemeProcessor;
import com.intellij.openapi.progress.ProgressManager;
import com.intellij.openapi.progress.impl.ProgressManagerImpl;
import com.intellij.openapi.projectRoots.ProjectJdkTable;
import com.intellij.openapi.projectRoots.Sdk;
import com.intellij.openapi.projectRoots.SdkType;
import com.intellij.openapi.projectRoots.impl.MockSdk;
import com.intellij.openapi.roots.AdditionalLibraryRootsProvider;
import com.intellij.openapi.roots.FileIndexFacade;
import com.intellij.openapi.roots.OrderRootType;
import com.intellij.openapi.roots.ProjectExtension;
import com.intellij.openapi.roots.ProjectFileIndex;
import com.intellij.openapi.roots.ProjectRootManager;
import com.intellij.openapi.roots.impl.DirectoryIndex;
import com.intellij.openapi.roots.impl.DirectoryIndexExcludePolicy;
import com.intellij.openapi.roots.impl.DirectoryIndexImpl;
import com.intellij.openapi.roots.impl.ProjectFileIndexImpl;
import com.intellij.openapi.roots.impl.ProjectRootManagerImpl;
import com.intellij.openapi.util.Disposer;
import com.intellij.openapi.util.StaticGetter;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.openapi.vfs.VirtualFileManager;
import com.intellij.openapi.vfs.VirtualFileSystem;
import com.intellij.openapi.vfs.encoding.EncodingManager;
import com.intellij.openapi.vfs.impl.VirtualFileManagerImpl;
import com.intellij.openapi.vfs.local.CoreLocalFileSystem;
import com.intellij.psi.PsiDocumentManager;
import com.intellij.psi.PsiElement;
import com.intellij.psi.PsiErrorElement;
import com.intellij.psi.PsiFile;
import com.intellij.psi.PsiFileFactory;
import com.intellij.psi.PsiManager;
import com.intellij.psi.SyntaxTraverser;
import com.intellij.psi.impl.DocumentCommitProcessor;
import com.intellij.psi.impl.DocumentCommitThread;
import com.intellij.psi.impl.PsiCachedValuesFactory;
import com.intellij.psi.impl.PsiDocumentManagerImpl;
import com.intellij.psi.impl.PsiFileFactoryImpl;
import com.intellij.psi.impl.PsiManagerImpl;
import com.intellij.psi.impl.PsiModificationTrackerImpl;
import com.intellij.psi.impl.file.PsiDirectoryFactory;
import com.intellij.psi.impl.file.PsiDirectoryFactoryImpl;
import com.intellij.psi.impl.source.resolve.ResolveCache;
import com.intellij.psi.search.ProjectScopeBuilder;
import com.intellij.psi.stubs.StubIndex;
import com.intellij.psi.stubs.StubIndexExtension;
import com.intellij.psi.stubs.StubIndexImpl;
import com.intellij.psi.util.CachedValuesManager;
import com.intellij.psi.util.PsiModificationTracker;
import com.intellij.util.CachedValuesManagerImpl;
import com.intellij.util.Function;
import com.intellij.util.QueryExecutor;
import com.intellij.util.containers.JBIterable;
import com.intellij.util.containers.MultiMap;
import com.intellij.util.indexing.FileBasedIndex;
import com.intellij.util.indexing.FileBasedIndexImpl;
import com.intellij.util.indexing.FileBasedIndexScanRunnableCollector;
import com.intellij.util.indexing.FileBasedIndexScanRunnableCollectorImpl;
import com.intellij.util.indexing.IndexableSetContributor;
import com.intellij.util.messages.MessageBus;
import com.jetbrains.numpy.codeInsight.NumpyDocStringTypeProvider;
import com.jetbrains.numpy.codeInsight.NumpyResolveRater;
import com.jetbrains.pyqt.PyQtTypeProvider;
import com.jetbrains.python.PyDirectoryIndexExcludePolicy;
import com.jetbrains.python.PythonDialectsTokenSetContributor;
import com.jetbrains.python.PythonFileType;
import com.jetbrains.python.PythonFileTypeFactory;
import com.jetbrains.python.PythonLanguage;
import com.jetbrains.python.PythonParserDefinition;
import com.jetbrains.python.PythonTokenSetContributor;
import com.jetbrains.python.codeInsight.stdlib.PyDataclassTypeProvider;
import com.jetbrains.python.codeInsight.stdlib.PyNamedTupleOverridingTypeProvider;
import com.jetbrains.python.codeInsight.stdlib.PyNamedTupleTypeProvider;
import com.jetbrains.python.codeInsight.stdlib.PyStdlibCanonicalPathProvider;
import com.jetbrains.python.codeInsight.stdlib.PyStdlibTypeProvider;
import com.jetbrains.python.codeInsight.typing.PyAncestorTypeProvider;
import com.jetbrains.python.codeInsight.typing.PyTypingTypeProvider;
import com.jetbrains.python.codeInsight.userSkeletons.PyUserSkeletonsTypeProvider;
import com.jetbrains.python.debugger.PyCallSignatureTypeProvider;
import com.jetbrains.python.debugger.PySignatureCacheManager;
import com.jetbrains.python.debugger.PySignatureCacheManagerImpl;
import com.jetbrains.python.documentation.docstrings.PyDocStringTypeProvider;
import com.jetbrains.python.psi.LanguageLevel;
import com.jetbrains.python.psi.PyCustomPackageIdentifier;
import com.jetbrains.python.psi.PyElementGenerator;
import com.jetbrains.python.psi.PyFile;
import com.jetbrains.python.psi.PyKnownDecoratorProvider;
import com.jetbrains.python.psi.PyPsiFacade;
import com.jetbrains.python.psi.impl.PyElementGeneratorImpl;
import com.jetbrains.python.psi.impl.PyImportResolver;
import com.jetbrains.python.psi.impl.PyPsiFacadeImpl;
import com.jetbrains.python.psi.impl.PyResolveResultRater;
import com.jetbrains.python.psi.impl.PyTypeProvider;
import com.jetbrains.python.psi.impl.references.PyReferenceCustomTargetChecker;
import com.jetbrains.python.psi.resolve.PyCanonicalPathProvider;
import com.jetbrains.python.psi.resolve.PyForwardReferenceResolveProvider;
import com.jetbrains.python.psi.resolve.PyReferenceResolveProvider;
import com.jetbrains.python.psi.resolve.PythonBuiltinReferenceResolveProvider;
import com.jetbrains.python.psi.resolve.PythonOverridingBuiltinReferenceResolveProvider;
import com.jetbrains.python.psi.search.PySuperMethodsSearchExecutor;
import com.jetbrains.python.psi.types.PyClassMembersProvider;
import com.jetbrains.python.psi.types.PyCollectionTypeByModificationsProvider;
import com.jetbrains.python.psi.types.PyModuleMembersProvider;
import com.jetbrains.python.psi.types.TypeEvalContext;
import com.jetbrains.python.psi.types.TypeEvalContextBasedCache;
import com.jetbrains.python.psi.types.TypeEvalContextCache;
import com.jetbrains.python.pyi.PyiFileTypeFactory;
import com.jetbrains.python.pyi.PyiLanguageDialect;
import com.jetbrains.python.pyi.PyiParserDefinition;
import com.jetbrains.python.pyi.PyiTypeProvider;
import com.jetbrains.python.sdk.PythonSdkType;
import com.jetbrains.python.sdk.flavors.PythonFlavorProvider;
import com.jetbrains.python.testing.PythonTestConfigurationType;
import com.jetbrains.python.testing.pyTestFixtures.PyTestFixtureTargetChecker;
import com.jetbrains.python.testing.pyTestParametrized.PyTestParametrizedTypeProvider;
import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import javax.annotation.Nullable;
import kotlin.jvm.functions.Function1;
import org.jetbrains.annotations.NonNls;
import org.jetbrains.annotations.NotNull;

public class PythonParser {

  private static final Disposable TEST_DISPOSABLE = new TestDisposable();
  private static final String SDK_DIR = "mySdk";

  private static final PsiFileFactory psiFileFactory = psiFileFactory();
  private static CoreLocalFileSystem fileSystem;

  public PyFile parse(String content) {
    PyFile file = parseAs(content, LanguageLevel.PYTHON38);
    if (errorElements(file).isEmpty()) {
      return file;
    }
    file = parseAs(content, LanguageLevel.PYTHON27);
    PsiErrorElement errorElement = errorElements(file).get(0);
    if (errorElement == null) {
      return file;
    }
    int lineNumber = new PythonTokenLocation(errorElement).startLine();
    throw new RecognitionException(lineNumber, errorElement.getErrorDescription());
  }

  private static JBIterable<PsiErrorElement> errorElements(PsiElement root) {
    return SyntaxTraverser.psiTraverser(root).traverse().filter(PsiErrorElement.class);
  }

  @NotNull
  private static PyFile parseAs(String content, LanguageLevel languageLevel) {
    PsiFile file = psiFileFactory.createFileFromText("test.py", PythonFileType.INSTANCE, normalizeEol(content), System.currentTimeMillis(), true, false);
    file.getViewProvider().getVirtualFile().putUserData(LanguageLevel.KEY, languageLevel);
    return (PyFile) file;
  }

  @NotNull
  public static String normalizeEol(String content) {
    return content.replaceAll("\\r\\n?", "\n");
  }


  public static PyFile parse(File file) {
    String fileContent;
    try {
      fileContent = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    return new PythonParser().parse(fileContent);
  }

  private static PsiFileFactory psiFileFactory() {
    File tmpDir;
    try {
      tmpDir = Files.createTempDirectory("sonar-python").toFile();
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    File sdkDir = initSdk(tmpDir);

    System.setProperty("idea.home.path", tmpDir.getAbsolutePath());
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
    application.registerService(PropertiesComponent.class, PropertiesComponentImpl.class);
    application.registerService(EncodingManager.class, CoreEncodingProjectManager.class);

    SchemeManagerFactory schemeManagerFactory = new SchemeManagerFactory() {
      @NotNull
      @Override
      public <SCHEME, MUTABLE_SCHEME extends SCHEME> SchemeManager<SCHEME> create(@NotNull String fileSpec, @NotNull SchemeProcessor<SCHEME, ? super MUTABLE_SCHEME> schemeProcessor, @Nullable String presentableName, @NotNull RoamingType roamingType, @NotNull Function1<? super String, String> schemeNameToFileName, @Nullable StreamProvider streamProvider, @Nullable Path ioDirectory, boolean b) {
        return (SchemeManager<SCHEME>) new EmptySchemesManager();
      }
    };
    application.registerService(SchemeManagerFactory.class, schemeManagerFactory);
    application.registerService(FileBasedIndex.class, FileBasedIndexImpl.class);
    CoreProjectJdkTable coreProjectJdkTable = new CoreProjectJdkTable();
    application.registerService(ProjectJdkTable.class, coreProjectJdkTable);

    ApplicationManager.setApplication(application, FileTypeRegistry.ourInstanceGetter, disposable);

    Extensions.registerAreaClass("IDEA_PROJECT", null);
    registerExtensionPoint(MetaLanguage.EP_NAME, MetaLanguage.class);
    registerExtensionPoint(PythonDialectsTokenSetContributor.EP_NAME, PythonDialectsTokenSetContributor.class);
    registerExtension(PythonDialectsTokenSetContributor.EP_NAME, new PythonTokenSetContributor());

    registerExtensionPoint(ConfigurationType.CONFIGURATION_TYPE_EP, ConfigurationType.class);
    registerExtension(ConfigurationType.CONFIGURATION_TYPE_EP, new PythonTestConfigurationType());

    registerExtensionPoint(IndexableSetContributor.EP_NAME, IndexableSetContributor.class);
    registerExtensionPoint(AdditionalLibraryRootsProvider.EP_NAME, AdditionalLibraryRootsProvider.class);

    registerExtensionPoint(PyTypeProvider.EP_NAME, PyTypeProvider.class);
    registerExtension(PyTypeProvider.EP_NAME, new PyStdlibTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyDataclassTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyNamedTupleTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyNamedTupleOverridingTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyCollectionTypeByModificationsProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyTypingTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyiTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyTestParametrizedTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyUserSkeletonsTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyCallSignatureTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new NumpyDocStringTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyDocStringTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyQtTypeProvider());
    registerExtension(PyTypeProvider.EP_NAME, new PyAncestorTypeProvider());

    registerExtensionPoint(FileTypeFactory.FILE_TYPE_FACTORY_EP, FileTypeFactory.class);
    registerExtension(PythonFileTypeFactory.FILE_TYPE_FACTORY_EP, new MyPythonFileTypeFactory());
    registerExtension(PythonFileTypeFactory.FILE_TYPE_FACTORY_EP, new PyiFileTypeFactory());

    registerExtensionPoint(PyReferenceCustomTargetChecker.Companion.getEP_NAME(), PyReferenceCustomTargetChecker.class);
    registerExtension(PyReferenceCustomTargetChecker.Companion.getEP_NAME(), new PyTestFixtureTargetChecker());

    registerExtensionPoint(PyReferenceResolveProvider.EP_NAME, PyReferenceResolveProvider.class);
    registerExtension(PyReferenceResolveProvider.EP_NAME, new PyForwardReferenceResolveProvider());
    registerExtension(PyReferenceResolveProvider.EP_NAME, new PythonBuiltinReferenceResolveProvider());
    registerExtension(PyReferenceResolveProvider.EP_NAME, new PythonOverridingBuiltinReferenceResolveProvider());

    registerExtensionPoint(PyImportResolver.EP_NAME, PyImportResolver.class);

    ExtensionPointName<QueryExecutor> superMethodSearch = ExtensionPointName.create("Pythonid.pySuperMethodsSearch");
    registerExtensionPoint(superMethodSearch, QueryExecutor.class);
    registerExtension(superMethodSearch, new PySuperMethodsSearchExecutor());

    // skipping extensions registration, limiting to extension point which is required
    registerExtensionPoint(PyClassMembersProvider.EP_NAME, PyClassMembersProvider.class);

    registerExtensionPoint(PyCustomPackageIdentifier.EP_NAME, PyCustomPackageIdentifier.class);
    registerExtensionPoint(PythonFlavorProvider.EP_NAME, PythonFlavorProvider.class);
    registerExtensionPoint(PyKnownDecoratorProvider.EP_NAME, PyKnownDecoratorProvider.class);

    registerExtensionPoint(PyCanonicalPathProvider.EP_NAME, PyCanonicalPathProvider.class);
    registerExtension(PyCanonicalPathProvider.EP_NAME, new PyStdlibCanonicalPathProvider());

    registerExtensionPoint(PyResolveResultRater.EP_NAME, PyResolveResultRater.class);
    registerExtension(PyResolveResultRater.EP_NAME, new NumpyResolveRater());

    // skipping extensions registration, limiting to extension point which is required
    // WARNING about PyStdlibModuleMembersProvider: it triggers loading of icon and start of UI
    registerExtensionPoint(PyModuleMembersProvider.EP_NAME, PyModuleMembersProvider.class);

    registerExtensionPoint(SdkType.EP_NAME, SdkType.class);
    registerExtension(SdkType.EP_NAME, getPythonSdkType());

    MockProject project = new MockProject(null, disposable);

    registerExtensionPoint(Extensions.getArea(project), ProjectExtension.EP_NAME, ProjectExtension.class);
    registerExtensionPoint(Extensions.getArea(project), PyDirectoryIndexExcludePolicy.EP_NAME, DirectoryIndexExcludePolicy.class);

    application.registerService(MessageBus.class, project.getMessageBus());
    FileTypeManagerImpl fileTypeManager = new FileTypeManagerImpl(project.getMessageBus(), schemeManagerFactory, new AppPropertiesComponentImpl());
    application.registerService(FileTypeManager.class, fileTypeManager);

    FileBasedIndexImpl fileBasedIndex = new FileBasedIndexImpl(null, fileDocMgr, fileTypeManager, project.getMessageBus());
    StubIndexImpl stubIndex = new StubIndexImpl(fileBasedIndex);
    application.registerService(StubIndex.class, stubIndex);

    // skipping extensions registration, limiting to extension point which is required
    Extensions.getRootArea().registerExtensionPoint(StubIndexExtension.EP_NAME, StubIndexExtension.class.getName(), ExtensionPoint.Kind.INTERFACE, TEST_DISPOSABLE);

    PsiModificationTrackerImpl modificationTracker = new PsiModificationTrackerImpl(project);
    project.registerService(PsiModificationTracker.class, modificationTracker);

    MockFileIndexFacade fileIndexFacade = new MockFileIndexFacade(project);
    project.registerService(FileIndexFacade.class, fileIndexFacade);

    project.registerService(DirectoryIndex.class, DirectoryIndexImpl.class);
    project.registerService(FileTypeRegistry.class, CoreFileTypeRegistry.class);
    project.registerService(ProjectFileIndex.class, ProjectFileIndexImpl.class);
    project.registerService(ResolveCache.class, new ResolveCache(project.getMessageBus()));

    ProjectRootManagerImpl projectRootManager = new ProjectRootManagerImpl(project);
    project.registerService(ProjectRootManager.class, projectRootManager);

    project.registerService(ProjectScopeBuilder.class, new CoreProjectScopeBuilder(project, fileIndexFacade));
    project.registerService(PySignatureCacheManager.class, PySignatureCacheManagerImpl.class);

    CoreModuleManager coreModuleManager = new CoreModuleManager(project, disposable);
    project.registerService(ModuleManager.class, coreModuleManager);

    project.registerService(FileTypeManager.class, fileTypeManager);
    project.registerService(FileBasedIndexScanRunnableCollector.class, FileBasedIndexScanRunnableCollectorImpl.class);
    project.registerService(SchemeManagerFactory.class, schemeManagerFactory);
    project.registerService(PyPsiFacade.class, new PyPsiFacadeImpl(project));
    project.registerService(PyElementGenerator.class, new PyElementGeneratorImpl(project));

    fileSystem = new CoreLocalFileSystem();
    VirtualFileSystem[] fs = new VirtualFileSystem[]{fileSystem};
    VirtualFileManagerImpl virtualFileManager = new VirtualFileManagerImpl(fs, project.getMessageBus());
    application.registerService(VirtualFileManager.class, virtualFileManager);

    // https://github.com/JetBrains/intellij-community/blob/93b632941e406178dd5c78fe4d8fdf7d8c357355/platform/testFramework/src/com/intellij/testFramework/ParsingTestCase.java
    //new MockPsiManager()
    PsiManagerImpl psiManager = new PsiManagerImpl(project, fileDocMgr, psiBuilderFactory, fileIndexFacade, project.getMessageBus(), modificationTracker);

    DocumentCommitProcessor documentCommitThread = getDocumentCommitThread(disposable);
    project.registerService(PsiDocumentManager.class, new PsiDocumentManagerImpl(project, psiManager, new EditorFactoryImpl(), project.getMessageBus(), documentCommitThread));

    project.registerService(PsiManager.class, psiManager);

    project.registerService(PsiDirectoryFactory.class, new PsiDirectoryFactoryImpl(psiManager));

    CachedValuesManagerImpl cachedValuesManager = new CachedValuesManagerImpl(project, new PsiCachedValuesFactory(psiManager));
    project.registerService(CachedValuesManager.class, cachedValuesManager);
    project.registerService(TypeEvalContextCache.class, new TypeEvalContextCacheImpl(cachedValuesManager));

    LanguageParserDefinitions.INSTANCE.addExplicitExtension(PythonLanguage.getInstance(), new PythonParserDefinition());
    LanguageParserDefinitions.INSTANCE.addExplicitExtension(PyiLanguageDialect.getInstance(), new PyiParserDefinition());
    CoreASTFactory astFactory = new CoreASTFactory();
    LanguageASTFactory.INSTANCE.addExplicitExtension(PythonLanguage.getInstance(), astFactory);
    LanguageASTFactory.INSTANCE.addExplicitExtension(Language.ANY, astFactory);

    PsiFileFactoryImpl psiFileFactory = new PsiFileFactoryImpl(psiManager);
    project.registerService(PsiFileFactory.class, psiFileFactory);

    Sdk sdk = createSDK(sdkDir.getAbsolutePath(), LanguageLevel.PYTHON38);
    coreProjectJdkTable.addJdk(sdk);
    projectRootManager.setProjectSdk(sdk);

    stubIndex.initComponent();

    return psiFileFactory;
  }

  @NotNull
  private static DocumentCommitProcessor getDocumentCommitThread(Disposable disposable) {
    Constructor<DocumentCommitThread> constructor2;
    DocumentCommitProcessor documentCommitThread;
    try {
      constructor2 = DocumentCommitThread.class.getDeclaredConstructor(ApplicationEx.class);
      constructor2.setAccessible(true);
      documentCommitThread = constructor2.newInstance(new MockApplicationEx(disposable));
    } catch (NoSuchMethodException | InstantiationException | IllegalAccessException | InvocationTargetException e) {
      throw new IllegalStateException(e);
    }
    return documentCommitThread;
  }

  @NotNull
  private static PythonSdkType getPythonSdkType() {
    Constructor<PythonSdkType> constructor;
    PythonSdkType pythonSdkType;
    try {
      constructor = PythonSdkType.class.getDeclaredConstructor();
      constructor.setAccessible(true);
      pythonSdkType = constructor.newInstance();
    } catch (NoSuchMethodException | InstantiationException | IllegalAccessException | InvocationTargetException e) {
      throw new IllegalStateException(e);
    }
    return pythonSdkType;
  }

  /**
   * Create our own PythonFileTypeFactory to skip loading of QtUIFileType and XmlFileType
   */
  private static class MyPythonFileTypeFactory extends FileTypeFactory {

    public void createFileTypes(@NonNls @NotNull FileTypeConsumer consumer) {
      consumer.consume(PythonFileType.INSTANCE, "py;pyw;");
    }
  }

  protected static <T> void registerExtension(@NotNull ExtensionPointName<T> extensionPointName, @NotNull T t) {
    registerExtension(Extensions.getRootArea(), extensionPointName, t);
  }

  public static <T> void registerExtension(@NotNull ExtensionsArea area, @NotNull ExtensionPointName<T> name, @NotNull T t) {
    registerExtensionPoint(area, name, (Class<T>) t.getClass());
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
      // Nothing to do
    }
  }

  static final class TypeEvalContextCacheImpl implements TypeEvalContextCache {

    @NotNull
    private static final Function<TypeEvalContext, TypeEvalContext> VALUE_PROVIDER = new MyValueProvider();
    @NotNull
    private final TypeEvalContextBasedCache<TypeEvalContext> myCache;

    TypeEvalContextCacheImpl(@NotNull final CachedValuesManager manager) {
      myCache = new TypeEvalContextBasedCache<>(manager, VALUE_PROVIDER);
    }


    @NotNull
    @Override
    public TypeEvalContext getContext(@NotNull final TypeEvalContext standard) {
      return myCache.getValue(standard);
    }

    private static class MyValueProvider implements Function<TypeEvalContext, TypeEvalContext> {

      @Override
      public TypeEvalContext fun(final TypeEvalContext param) {
        // key and value are both context here. If no context stored, then key is stored. Old one is returned otherwise to cache.
        return param;
      }
    }
  }

  private static Sdk createSDK(String sdkDir, LanguageLevel level) {
    File pythonBin = new File(sdkDir, "bin/python" + level);
    createFile(pythonBin);

    MultiMap<OrderRootType, VirtualFile> roots = MultiMap.create();

    VirtualFile typeShedDir = fileSystem.findFileByPath(sdkDir + "/typeshed");
    roots.putValue(OrderRootType.CLASSES, typeShedDir.findFileByRelativePath("2and3"));

    MockSdk sdk = new MockSdk("Mock Python SDK", pythonBin.getPath(), "Python " + level + " Mock SDK", roots, PythonSdkType.getInstance());

    // com.jetbrains.python.psi.resolve.PythonSdkPathCache.getInstance() corrupts SDK, so have to clone
    return sdk.clone();
  }

  private static File initSdk(File workDir) {
    File sdkDir = new File(workDir + "/" + SDK_DIR);
    createDir(sdkDir);
    createDir(new File(sdkDir, "/bin"));

    File typeshed = new File(sdkDir, "typeshed");
    createDir(typeshed);

    File python2and3 = new File(typeshed, "2and3");
    createDir(python2and3);
    copyResource("/org/sonar/python/mySdk/typeshed/2and3/argparse.pyi", new File(python2and3, "argparse.pyi").getAbsolutePath());
    copyResource("/org/sonar/python/mySdk/typeshed/2and3/optparse.pyi", new File(python2and3, "optparse.pyi").getAbsolutePath());
    copyResource("/org/sonar/python/mySdk/typeshed/2and3/sys.pyi", new File(python2and3, "sys.pyi").getAbsolutePath());
    return sdkDir;
  }

  private static void copyResource(String res, String dest) {
    InputStream src = PythonParser.class.getResourceAsStream(res);
    try {
      Files.copy(src, Paths.get(dest), StandardCopyOption.REPLACE_EXISTING);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  private static void createFile(File file) {
    try {
      if (!file.createNewFile()) {
        throw new IllegalStateException("Unable to create file: " + file);
      }
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  private static void createDir(File file) {
    if (!file.mkdir()) {
      throw new IllegalStateException("Unable to create dir: " + file);
    }
  }
}
