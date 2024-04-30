from django import forms

class UploadForm(forms.Form):
    file = forms.FileField()

class UploadFileForm(forms.Form):
    file = forms.FileField()

# forms.py




class UploadDatasetForm(forms.Form):
    dataset = forms.FileField(label='Upload Dataset', help_text='Supported formats: CSV, Excel')

class ModelForm(forms.Form):
    target_label = forms.CharField(label='Select Target Label')
    remove_columns = forms.MultipleChoiceField(label='Columns to Remove', required=False, widget=forms.CheckboxSelectMultiple)
    model_type = forms.ChoiceField(label='Select Model Type', choices=[])


#class ModelForm__(forms.Form):
#    dataset = forms.FileField(label='Upload Dataset')
#    target_column_name = forms.CharField(label='Target Column Name')


class ModelSelectForm(forms.Form):
    MODELS_CHOICES = [
        ('logistic_regression', 'Logistic Regression'),
        ('decision_tree', 'Decision Tree'),
        # Add more choices for other models
    ]
    selected_model = forms.ChoiceField(choices=MODELS_CHOICES, label='Select Model')


from django import forms

class TargetColumnForm(forms.Form):
    target_column_name = forms.CharField(label='Target Column Name')
