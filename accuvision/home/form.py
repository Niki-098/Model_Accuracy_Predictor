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





# forms.py



class TargetColumnForm(forms.Form):
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    dataset_file = forms.FileField(label='Upload Dataset', help_text='Accepts .csv, .xlsx, .xls files')


class LinearRegressionForm(forms.Form):
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    dataset_file = forms.FileField(label='Upload Dataset', help_text='Accepts .csv, .xlsx, .xls files')


class DecisionTreeForm(forms.Form):
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    dataset_file = forms.FileField(label='Upload Dataset', help_text='Accepts .csv, .xlsx, .xls files')


class RandomForestForm(forms.Form):
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    dataset_file = forms.FileField(label='Upload Dataset', help_text='Accepts .csv, .xlsx, .xls files')

class GradientBoostingForm(forms.Form):
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    dataset_file = forms.FileField(label='Upload Dataset', help_text='Accepts .csv, .xlsx, .xls files')


class SVMForm(forms.Form):
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    dataset_file = forms.FileField(label='Upload Dataset', help_text='Accepts .csv, .xlsx, .xls files')


class KNNForm(forms.Form):
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    dataset_file = forms.FileField(label='Upload Dataset', help_text='Accepts .csv, .xlsx, .xls files')


class Naive_Bayes_Form(forms.Form):
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    dataset_file = forms.FileField(label='Upload Dataset', help_text='Accepts .csv, .xlsx, .xls files')




class Kmeans_clustering_Form(forms.Form):
    dataset_file = forms.FileField(label='Upload Dataset')
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    num_clusters = forms.IntegerField(label='Number of Clusters', min_value=2, initial=2)



class hierarchical_clustering_Form(forms.Form):
    dataset_file = forms.FileField(label='Upload Dataset')
    target_column_name = forms.CharField(label='Target Column Name', max_length=100)
    num_clusters = forms.IntegerField(label='Number of Clusters', min_value=2, initial=2)


