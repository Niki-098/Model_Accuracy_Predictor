from django import forms

class UploadForm(forms.Form):
    file = forms.FileField()

class UploadFileForm(forms.Form):
    file = forms.FileField()

# forms.py








class ModelForm(forms.Form):
    target_label = forms.ChoiceField(choices=[])
    remove_columns = forms.MultipleChoiceField(choices=[])
    
    def __init__(self, *args, dataset_columns=None, **kwargs):
        super(ModelForm, self).__init__(*args, **kwargs)
        if dataset_columns is not None:
            self.fields['target_label'].choices = [(col, col) for col in dataset_columns]
            self.fields['remove_columns'].choices = [(col, col) for col in dataset_columns]
 