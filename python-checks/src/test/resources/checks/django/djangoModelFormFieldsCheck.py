from django import forms

class ExcludeFieldsForm(forms.ModelForm):
    class Meta:
        model = MyModel
        exclude = ['field1', 'field2']  # Noncompliant {{Set the fields of this form explicitly instead of using "exclude".}}

class AllFieldsForm(forms.ModelForm):
    class Meta:
        model = MyModel
        fields = '__all__'  # Noncompliant {{Set the fields of this form explicitly instead of using "__all__".}}

class MissingFieldsForm(forms.ModelForm):
    class Meta:
        model = MyModel

class MissingMetaForm(forms.ModelForm):
    ...


class CompliantForm(forms.ModelForm):
    class Meta:
        model = MyModel
        fields = ['field1', 'field2']

class CompliantForm(forms.ModelForm):
    class Meta:
        model = MyModel
        fields = 'field1'
