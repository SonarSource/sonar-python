/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.semantic;

import java.io.File;
import java.net.URI;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeShedPythonFile;

public class SymbolUtils {

  private static final String SEND_MESSAGE = "send_message";
  private static final String SET_COOKIE = "set_cookie";
  private static final String SET_SIGNED_COOKIE = "set_signed_cookie";
  private static final String EQ = "__eq__";
  private static final String SET_VERIFY = "set_verify";

  private SymbolUtils() {
  }

  public static String fullyQualifiedModuleName(String packageName, String fileName) {
    int extensionIndex = fileName.lastIndexOf('.');
    String moduleName = extensionIndex > 0
      ? fileName.substring(0, extensionIndex)
      : fileName;
    if (moduleName.equals("__init__")) {
      return packageName;
    }
    return packageName.isEmpty()
      ? moduleName
      : (packageName + "." + moduleName);
  }

  public static Set<Symbol> globalSymbols(FileInput fileInput, String packageName, PythonFile pythonFile) {
    SymbolTableBuilder symbolTableBuilder = new SymbolTableBuilder(packageName, pythonFile);
    String fullyQualifiedModuleName = SymbolUtils.fullyQualifiedModuleName(packageName, pythonFile.fileName());
    fileInput.accept(symbolTableBuilder);
    Set<Symbol> globalSymbols = new HashSet<>();
    for (Symbol globalVariable : fileInput.globalVariables()) {
      String fullyQualifiedVariableName = globalVariable.fullyQualifiedName();
      if (((fullyQualifiedVariableName != null) && !fullyQualifiedVariableName.startsWith(fullyQualifiedModuleName)) ||
        globalVariable.usages().stream().anyMatch(u -> u.kind().equals(Usage.Kind.IMPORT))) {
        // TODO: We don't put builtin or imported names in global symbol table to avoid duplicate FQNs in project level symbol table (to fix with SONARPY-647)
        continue;
      }
      if (globalVariable.kind() == Symbol.Kind.CLASS) {
        globalSymbols.add(((ClassSymbolImpl) globalVariable).copyWithoutUsages());
      } else if (globalVariable.kind() == Symbol.Kind.FUNCTION) {
        globalSymbols.add(new FunctionSymbolImpl(globalVariable.name(), ((FunctionSymbol) globalVariable)));
      } else {
        globalSymbols.add(new SymbolImpl(globalVariable.name(), fullyQualifiedModuleName + "." + globalVariable.name()));
      }
    }
    return globalSymbols;
  }

  static void resolveTypeHierarchy(ClassDef classDef, @Nullable Symbol symbol, PythonFile pythonFile, Map<String, Symbol> symbolsByName) {
    if (symbol == null || !Symbol.Kind.CLASS.equals(symbol.kind())) {
      return;
    }
    ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
    if (isBuiltinTypeshedFile(pythonFile) && "str".equals(classSymbol.fullyQualifiedName())) {
      classSymbol.addSuperClass(symbolsByName.get("object"));
      classSymbol.addSuperClass(symbolsByName.get("Sequence"));
      return;
    }
    ArgList argList = classDef.args();
    if (argList == null) {
      return;
    }
    for (Argument argument : argList.arguments()) {
      Symbol argumentSymbol = getSymbolFromArgument(argument);
      if (argumentSymbol == null) {
        classSymbol.setHasSuperClassWithoutSymbol();
      } else {
        Symbol normalizedArgumentSymbol = normalizeSymbol(argumentSymbol, pythonFile, symbolsByName);
        if (normalizedArgumentSymbol != null) {
          classSymbol.addSuperClass(normalizedArgumentSymbol);
        }
      }
    }
  }

  /**
   * Hardcoding some 'typing' module symbols to avoid incomplete type hierarchy for type 'str'
   */
  @CheckForNull
  private static Symbol normalizeSymbol(Symbol symbol, PythonFile pythonFile, Map<String, Symbol> symbolsByName) {
    if (isTypeShedFile(pythonFile) && (symbol.name().equals("Protocol") || symbol.name().equals("Generic"))) {
      // ignore Protocol and Generic to avoid having incomplete type hierarchies
      return null;
    }
    if (isTypingFile(pythonFile) && symbol.name().equals("_Collection")) {
      return symbolsByName.get("Collection");
    }
    return symbol;
  }

  private static boolean isBuiltinTypeshedFile(PythonFile pythonFile) {
    return isTypeShedFile(pythonFile) && pythonFile.fileName().isEmpty();
  }

  private static boolean isTypingFile(PythonFile pythonFile) {
    return isTypeShedFile(pythonFile) && pythonFile.fileName().equals("typing");
  }

  @CheckForNull
  private static Symbol getSymbolFromArgument(Argument argument) {
    if (argument.is(Kind.REGULAR_ARGUMENT)) {
      Expression expression = ((RegularArgument) argument).expression();
      while (expression.is(Kind.SUBSCRIPTION)) {
        // to support using 'typing' symbols like 'List[str]'
        expression = ((SubscriptionExpression) expression).object();
      }
      if (expression instanceof HasSymbol) {
        return ((HasSymbol) expression).symbol();
      }
    }
    return null;
  }

  public static List<Expression> assignmentsLhs(AssignmentStatement assignmentStatement) {
    return assignmentStatement.lhsExpressions().stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .flatMap(TreeUtils::flattenTuples)
      .collect(Collectors.toList());
  }

  static List<Name> boundNamesFromExpression(@CheckForNull Tree tree) {
    List<Name> names = new ArrayList<>();
    if (tree == null) {
      return names;
    }
    if (tree.is(Tree.Kind.NAME)) {
      names.add(((Name) tree));
    } else if (tree.is(Tree.Kind.TUPLE)) {
      ((Tuple) tree).elements().forEach(t -> names.addAll(boundNamesFromExpression(t)));
    } else if (tree.is(Kind.LIST_LITERAL)) {
      ((ListLiteral) tree).elements().expressions().forEach(t -> names.addAll(boundNamesFromExpression(t)));
    } else if (tree.is(Kind.PARENTHESIZED)) {
      names.addAll(boundNamesFromExpression(((ParenthesizedExpression) tree).expression()));
    } else if (tree.is(Kind.UNPACKING_EXPR)) {
      names.addAll(boundNamesFromExpression(((UnpackingExpression) tree).expression()));
    }
    return names;
  }

  public static String pythonPackageName(File file, File projectBaseDir) {
    File currentDirectory = file.getParentFile();
    Deque<String> packages = new ArrayDeque<>();
    while (!currentDirectory.getAbsolutePath().equals(projectBaseDir.getAbsolutePath())) {
      File initFile = new File(currentDirectory, "__init__.py");
      if (!initFile.exists()) {
        break;
      }
      packages.push(currentDirectory.getName());
      currentDirectory = currentDirectory.getParentFile();
    }
    return String.join(".", packages);
  }

  @CheckForNull
  static Path pathOf(PythonFile pythonFile) {
    try {
      URI uri = pythonFile.uri();
      if ("file".equalsIgnoreCase(uri.getScheme())) {
        return Paths.get(uri);
      }
      return null;
    } catch (InvalidPathException e) {
      return null;
    }
  }

  public static Map<String, Set<Symbol>> externalModulesSymbols() {
    Map<String, Set<Symbol>> globalSymbols = new HashMap<>();

    globalSymbols.put("flask_mail", new HashSet<>(Arrays.asList(
      classSymbol("Mail", "flask_mail.Mail", "send", SEND_MESSAGE),
      classSymbol("Connection", "flask_mail.Connection", "send", SEND_MESSAGE)
      )));
    globalSymbols.put("smtplib", new HashSet<>(Arrays.asList(
      classSymbol("SMTP", "smtplib.SMTP", "sendmail", SEND_MESSAGE, "starttls"),
      classSymbol("SMTP_SSL", "smtplib.SMTP_SSL", "sendmail", SEND_MESSAGE)
    )));
    globalSymbols.put("http.cookies", new HashSet<>(Collections.singletonList(classSymbol("SimpleCookie", "http.cookies.SimpleCookie"))));

    globalSymbols.put("django.http", new HashSet<>(Arrays.asList(
      classSymbol("HttpResponse", "django.http.HttpResponse", SET_COOKIE, SET_SIGNED_COOKIE, "__setitem__"),
      classSymbol("HttpResponseRedirect", "django.http.HttpResponseRedirect", SET_COOKIE, SET_SIGNED_COOKIE),
      classSymbol("HttpResponsePermanentRedirect", "django.http.HttpResponsePermanentRedirect", SET_COOKIE, SET_SIGNED_COOKIE),
      classSymbol("HttpResponseNotModified", "django.http.HttpResponseNotModified", SET_COOKIE, SET_SIGNED_COOKIE),
      classSymbol("HttpResponseNotFound", "django.http.HttpResponseNotFound", SET_COOKIE, SET_SIGNED_COOKIE),
      classSymbol("HttpResponseForbidden", "django.http.HttpResponseForbidden", SET_COOKIE, SET_SIGNED_COOKIE),
      classSymbol("HttpResponseNotAllowed", "django.http.HttpResponseNotAllowed", SET_COOKIE, SET_SIGNED_COOKIE),
      classSymbol("HttpResponseGone", "django.http.HttpResponseGone", SET_COOKIE, SET_SIGNED_COOKIE),
      classSymbol("HttpResponseServerError", "django.http.HttpResponseServerError", SET_COOKIE, SET_SIGNED_COOKIE),
      classSymbol("HttpResponseBadRequest", "django.http.HttpResponseBadRequest", SET_COOKIE, SET_SIGNED_COOKIE)
    )));

    globalSymbols.put("django.http.response", new HashSet<>(Collections.singleton(
      classSymbol("HttpResponse", "django.http.response.HttpResponse")
    )));

    ClassSymbolImpl flaskResponse = classSymbol("Response", "flask.Response", SET_COOKIE);

    FunctionSymbolImpl makeResponse = new FunctionSymbolImpl("make_response", "flask.make_response", false, false, false, Collections.emptyList(),
      Collections.emptyList());
    makeResponse.setDeclaredReturnType(InferredTypes.runtimeType(flaskResponse));

    FunctionSymbolImpl redirect = new FunctionSymbolImpl("redirect", "flask.redirect", false, false, false, Collections.emptyList(),
       Collections.emptyList());
    redirect.setDeclaredReturnType(InferredTypes.runtimeType(flaskResponse));

    globalSymbols.put("flask", new HashSet<>(Arrays.asList(
      flaskResponse,
      makeResponse,
      redirect
    )));

    globalSymbols.put("werkzeug.datastructures", new HashSet<>(Collections.singleton(
      classSymbol("Headers", "werkzeug.datastructures.Headers", "set", "setdefault", "__setitem__")
    )));

    // TODO To be removed once we import 'collections' from typeshed
    globalSymbols.put("collections", new HashSet<>(Arrays.asList(
      classSymbol("deque", "collections.deque", EQ),
      classSymbol("UserList", "collections.UserList", EQ),
      classSymbol("UserDict", "collections.UserDict", EQ),
      classSymbol("ChainMap", "collections.ChainMap", EQ),
      classSymbol("Counter", "collections.Counter", EQ),
      classSymbol("OrderedDict", "collections.OrderedDict", EQ),
      classSymbol("defaultdict", "collections.defaultdict", EQ)
    )));


    ClassSymbolImpl ldapObject = classSymbol("LDAPObject", "ldap.LDAPObject", "simple_bind", "simple_bind_s", "bind", "bind_s");
    FunctionSymbolImpl initialize = new FunctionSymbolImpl(
      "initialize", "ldap.initialize", false, false, false, Collections.emptyList(),Collections.emptyList());
    initialize.setDeclaredReturnType(InferredTypes.runtimeType(ldapObject));
    globalSymbols.put("ldap", new HashSet<>(Collections.singleton(initialize)));


    ClassSymbolImpl sslContextClass =
      classSymbol("Context", "OpenSSL.SSL.Context", SET_VERIFY);
    SymbolImpl sslSubmodule = moduleSymbol("SSL", "OpenSSL.SSL", sslContextClass);
    globalSymbols.put("OpenSSL", Collections.singleton(sslSubmodule));


    ClassSymbolImpl csrfProtect =
      classSymbol("CSRFProtect", "flask_wtf.csrf.CSRFProtect", "init_app", "exempt");
    globalSymbols.put("flask_wtf.csrf", Collections.singleton(csrfProtect));


    ClassSymbolImpl modesCBC = classSymbol("CBC", "cryptography.hazmat.primitives.ciphers.modes.CBC");
    ClassSymbolImpl modesECB = classSymbol("ECB", "cryptography.hazmat.primitives.ciphers.modes.ECB");
    SymbolImpl cryptographyModesSubmodule = moduleSymbol("modes", "cryptography.hazmat.primitives.ciphers.modes", modesCBC, modesECB);
    globalSymbols.put("cryptography.hazmat.primitives.ciphers", Collections.singleton(cryptographyModesSubmodule));

    ClassSymbolImpl pkcs1v15 = classSymbol("PKCS1v15", "cryptography.hazmat.primitives.asymmetric.padding.PKCS1v15");
    SymbolImpl cryptographyPaddingSubmodule = moduleSymbol("padding", "cryptography.hazmat.primitives.asymmetric.padding", pkcs1v15);

    FunctionSymbolImpl generatePrivateKey = new FunctionSymbolImpl(
      "generate_private_key", "cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key", false, false, false, Collections.emptyList(),Collections.emptyList());
    ClassSymbolImpl rsaPrivateKey = classSymbol("RSAPrivateKey", "cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey", "decrypt");
    FunctionSymbolImpl publicKey = new FunctionSymbolImpl(
      "public_key", "cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey.public_key", false, false, false, Collections.emptyList(),Collections.emptyList());
    ClassSymbolImpl rsaPublicKey = classSymbol("RSAPublicKey", "cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey", "encrypt");
    publicKey.setDeclaredReturnType(InferredTypes.runtimeType(rsaPublicKey));
    rsaPrivateKey.addMembers(Collections.singleton(publicKey));
    generatePrivateKey.setDeclaredReturnType(InferredTypes.runtimeType(rsaPrivateKey));
    SymbolImpl cryptographyRsaSubmodule = moduleSymbol("rsa", "cryptography.hazmat.primitives.asymmetric.rsa", generatePrivateKey);

    globalSymbols.put("cryptography.hazmat.primitives.asymmetric", new HashSet<>(Arrays.asList(cryptographyPaddingSubmodule, cryptographyRsaSubmodule)));

    return globalSymbols;
  }

  private static ClassSymbolImpl classSymbol(String name, String fullyQualifiedName, String... members) {
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(name, fullyQualifiedName);
    classSymbol.addMembers(Arrays.stream(members).map(m -> new SymbolImpl(m, fullyQualifiedName + "." + m)).collect(Collectors.toSet()));
    return classSymbol;
  }

  @SuppressWarnings("SameParameterValue")
  private static SymbolImpl moduleSymbol(String moduleName, String fullyQualifiedName, Symbol... childSymbols) {
    SymbolImpl m = new SymbolImpl(moduleName, fullyQualifiedName);
    for (Symbol c: childSymbols) {
      m.addChildSymbol(c);
    }
    return m;
  }

  public static boolean isTypeShedFile(PythonFile pythonFile) {
    return pythonFile instanceof TypeShedPythonFile;
  }

  /**
   * @return the offset between parameter position and argument position:
   *   0 if there is no implicit first parameter (self, cls, etc...)
   *   1 if there is an implicit first parameter
   *  -1 if unknown (intent is not clear from context)
   */
  public static int firstParameterOffset(FunctionSymbol functionSymbol, boolean isStaticCall) {
    List<FunctionSymbol.Parameter> parameters = functionSymbol.parameters();
    if (parameters.isEmpty()) {
      return 0;
    }
    String firstParamName = parameters.get(0).name();
    if (firstParamName == null) {
      // First parameter is defined as a tuple
      return -1;
    }
    List<String> decoratorNames = functionSymbol.decorators();
    if (decoratorNames.size() > 1) {
      // We want to avoid FP if there are many decorators
      return -1;
    }
    if (!decoratorNames.isEmpty() && !decoratorNames.get(0).endsWith("classmethod") && !decoratorNames.get(0).endsWith("staticmethod")) {
      // Unknown decorator which might alter the behaviour of the method
      return -1;
    }
    if (functionSymbol.isInstanceMethod() && !isStaticCall) {
      // regular instance call, takes self as first implicit parameter
      return 1;
    }
    if (decoratorNames.size() == 1 && decoratorNames.get(0).endsWith("classmethod")) {
      // class method call, takes cls as first implicit parameter
      return 1;
    }
    // regular static call (function or method), no first implicit parameter
    return 0;
  }
}
