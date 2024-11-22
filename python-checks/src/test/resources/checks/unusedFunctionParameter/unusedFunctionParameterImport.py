from unusedFunctionParameterImported import ImportedParent, ParentWithDuplicatedParent, MyClassWithAnnotatedMember

class ChildFromImported(ImportedParent):

    # SONARPY-1829 `method_defined_in_child_class_only` is not considered a member of ImportedParent class, thus S1172 is raised
    def method_defined_in_child_class_only(self, a): # Noncompliant
        #                                        ^
        return compute()

    def method_defined_in_child_class_only_and_not_used(self, a): # Noncompliant
        #                                                     ^
        return compute()

class ChildFromDuplicated(ParentWithDuplicatedParent):

    def do_something(self, a): # FN SONARPY-1829 ChildFromDuplicated has an unresolved type hierarchy, because of the duplicated parent classes
        return compute()

class MyChild(MyClassWithAnnotatedMember):
    def my_member(self, param, other_param): # OK, respecting contract defined in parent
        return compute()
