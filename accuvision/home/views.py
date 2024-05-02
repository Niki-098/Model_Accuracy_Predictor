import os
from django.conf import settings
from django.forms import ModelForm
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import render, redirect
import pandas as pd
from .form import GradientBoostingForm, KNNForm, UploadForm, UploadFileForm
from .models import UploadedFile
from django.http import JsonResponse
from django.contrib.auth.models import User,auth

def SignUp(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
 
        data = User.objects.create_user(first_name = first_name, last_name = last_name, email=email,username=username, password=password1)
        data.save()

    return render(request,'login.html')


def Login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(username=username,password=password)

        if user is not None:
            auth.login(request,user)
            return redirect('index1.html')
        
    return render(request,'login.html')

# Create your views here.


def home1(request):
    return render(request, 'index1.html')


def home(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html') 


def upload_file(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            # Save the uploaded file to the appropriate location
            with open(os.path.join(settings.MEDIA_ROOT, uploaded_file.name), 'wb') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
                #Render the success page with the uploaded file name
            return render(request, 'upload_success.html', {'filename': uploaded_file.name})
    else:
        form = UploadForm()
    return render(request, 'upload_form.html', {'form': form})


def upload_success(request, filename):
    # Pass uploaded file information to the template
    context = {'filename': filename}
    return render(request, 'upload_success.html', context)



def show_uploaded_files(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('show_uploaded_files')
    else:
        form = UploadForm()

    # Retrieve all uploaded files from the database
    files = UploadedFile.objects.all()
    return render(request, 'uploaded_files.html', {'files': files, 'form': form})


from django.http import HttpResponseRedirect
from .utils import auto_preprocess_dataset, train_gradient_boosting, train_knn

def upload_files(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Handle uploaded file here
            new_file = UploadedFile(file=request.FILES['file'])
            new_file.save()
            return HttpResponseRedirect('/preprocess/')
    else:
        form = UploadFileForm()
    return render(request, 'upload_files.html', {'form': form})

from .models import UploadedFile  # Import the model where uploaded files are stored

def preprocess_files(request):
    # Fetch all uploaded files by the current user
    files = UploadedFile.objects.filter(uploaded_by=request.user)
    return render(request, 'preprocess.html', {'files': files})





from django.urls import reverse

def preprocessing_files(request):
    if request.method == 'POST':
        # Get the selected file id from the form
        selected_file_id = request.POST.get('selected_file')
        selected_file = UploadedFile.objects.get(id=selected_file_id)
        
        # Read the selected dataset into pandas DataFrame
        dataset = pd.read_csv(selected_file.file)
        
        # Preprocess the dataset
        preprocessed_data = auto_preprocess_dataset(dataset)
        #context = {'preprocessed_data': preprocessed_data}
        #return render(request, 'preprocessed_files.html', context)
            # Call download_preprocessed_data and pass preprocessed_data as a parameter
        preprocessed_data = auto_preprocess_dataset(dataset)
        #context = {'preprocessed_data': preprocessed_data}  # Pass preprocessed_data to the template context
        #return render(request, 'preprocessed_files.html', context)
        # Store the preprocessed data in the session
        request.session['preprocessed_data'] = preprocessed_data.to_json()
        
        # Redirect to the preprocessed files page
        return redirect('preprocessed_files')
    else:
        pass
        

import io

def download_preprocessed_data(request):
    # Retrieve the preprocessed data from the session
    preprocessed_data_json = request.session.get('preprocessed_data', None)
    
    if preprocessed_data_json:
        # Convert the JSON data back to a DataFrame
        preprocessed_data = pd.read_json(preprocessed_data_json)
        
        # Create a BytesIO object to store the Excel file
        excel_file = io.BytesIO()
        
        # Use the BytesIO object as the excel_writer argument
        preprocessed_data.to_excel(excel_file, index=False)

        # Prepare the response with the Excel data
        response = HttpResponse(excel_file.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename="preprocessed_data.xlsx"'
        return response
    else:
        # If preprocessed data is not found, handle the error appropriately
        return HttpResponse("Preprocessed data not found.", status=404)





# views.py

from .form import UploadDatasetForm, ModelForm

def extract_column_names(file):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file)

    # Get the column names
    column_names = df.columns.tolist()

    return column_names

import pandas as pd

def model_page(request):
    if request.method == 'POST':
        form = ModelForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.cleaned_data['dataset']
            remove_columns = request.POST.get('remove_columns', '').split(',')  # Get columns to remove
            target_label = request.POST.get('target_label', '')  # Get target label column name
            
            # Read the uploaded dataset into a pandas DataFrame
            df = pd.read_excel(dataset)
            
            # Remove the specified columns
            df.drop(remove_columns, axis=1, inplace=True)
            
            # Now you can use 'df' for further processing
            
            return render(request, 'model.html', {'form': form})
    else:
        form = ModelForm()
    return render(request, 'model.html', {'form': form})




def model_selection(request):
    if request.method == 'POST':
        # Handle form submission
        dataset = request.FILES['dataset']
        remove_columns = request.POST.get('remove_columns').split(',')
        target_label = request.POST.get('target_label')
        
        # Read the dataset into a pandas DataFrame
        df = pd.read_excel(dataset)
        
        # Remove specified columns
        df.drop(columns=remove_columns, inplace=True)
        
        # Export the modified DataFrame to Excel
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False)
        writer._save()
        output.seek(0)
        
        # Prepare response with the modified Excel file for download
        response = HttpResponse(output.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename=modified_dataset.xlsx'
        return response
        
    return render(request, 'model.html')

def model_select_view(request):
    return render(request, 'model_select.html')


from .utils import train_logistic_regression
#from .form import ModelForm__

from django.shortcuts import render, redirect
from .form import TargetColumnForm
from .utils import train_logistic_regression

import pandas as pd

from django.shortcuts import render

from django.shortcuts import render
from django.http import HttpResponseBadRequest
from .utils import train_logistic_regression

def logistic_regression_view(request):
    if request.method == 'POST':
        # Check if the form contains both dataset and target column name
        form = TargetColumnForm(request.POST, request.FILES)
        if form.is_valid():
            # Extract the target column name from the form
            target_column_name = form.cleaned_data['target_column_name']

            # Process the uploaded dataset
            dataset_file = form.cleaned_data['dataset_file']
            if dataset_file.name.endswith('.csv'):
                dataset = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith('.xlsx') or dataset_file.name.endswith('.xls'):
                dataset = pd.read_excel(dataset_file)
            else:
                # Handle unsupported file formats or raise an error
                return HttpResponseBadRequest("Unsupported file format")

            # Call the logistic regression training function
            accuracy = train_logistic_regression(dataset, target_column_name)

            # Render the results template with the accuracy
            return render(request, 'results.html', {'accuracy': accuracy})
    else:
        form = TargetColumnForm()
    return render(request, 'logistic_regression.html', {'form': form})



from django.http import HttpResponseBadRequest
from .form import DecisionTreeForm
from .utils import train_decision_tree
import pandas as pd

def decision_tree_view(request):
    if request.method == 'POST':
        # Check if the form contains both dataset and target column name
        form = DecisionTreeForm(request.POST, request.FILES)
        if form.is_valid():
            # Extract the target column name from the form
            target_column_name = form.cleaned_data['target_column_name']

            # Process the uploaded dataset
            dataset_file = form.cleaned_data['dataset_file']
            if dataset_file.name.endswith('.csv'):
                dataset = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith('.xlsx') or dataset_file.name.endswith('.xls'):
                dataset = pd.read_excel(dataset_file)
            else:
                # Handle unsupported file formats or raise an error
                return HttpResponseBadRequest("Unsupported file format")

            # Call the decision tree training function
            accuracy = train_decision_tree(dataset, target_column_name)

            # Render the results template with the accuracy
            return render(request, 'decision_tree_results.html', {'accuracy': accuracy})
    else:
        form = DecisionTreeForm()
    return render(request, 'decision_tree.html', {'form': form})




from .form import RandomForestForm
from .utils import train_random_forest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def random_forest_view(request):
    if request.method == 'POST':
        # Check if the form contains both dataset and target column name
        form = RandomForestForm(request.POST, request.FILES)
        if form.is_valid():
            # Extract the target column name from the form
            target_column_name = form.cleaned_data['target_column_name']

            # Process the uploaded dataset
            dataset_file = form.cleaned_data['dataset_file']
            if dataset_file.name.endswith('.csv'):
                dataset = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith('.xlsx') or dataset_file.name.endswith('.xls'):
                dataset = pd.read_excel(dataset_file)
            else:
                # Handle unsupported file formats or raise an error
                return HttpResponseBadRequest("Unsupported file format")

            # Call the random forest training function
            accuracy = train_random_forest(dataset, target_column_name)

            # Render the results template with the accuracy
            return render(request, 'random_forest_results.html', {'accuracy': accuracy})
    else:
        form = RandomForestForm()
    return render(request, 'random_forest.html', {'form': form})


def gradient_boosting_view(request):
    if request.method == 'POST':
        # Check if the form contains both dataset and target column name
        form = GradientBoostingForm(request.POST, request.FILES)
        if form.is_valid():
            # Extract the target column name from the form
            target_column_name = form.cleaned_data['target_column_name']

            # Process the uploaded dataset
            dataset_file = form.cleaned_data['dataset_file']
            if dataset_file.name.endswith('.csv'):
                dataset = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith('.xlsx') or dataset_file.name.endswith('.xls'):
                dataset = pd.read_excel(dataset_file)
            else:
                # Handle unsupported file formats or raise an error
                return HttpResponseBadRequest("Unsupported file format")

            # Call the GBM regression training function
            accuracy = train_gradient_boosting(dataset, target_column_name)

            # Render the results template with the accuracy
            return render(request, 'gradient_boosting_results.html', {'accuracy': accuracy})
    else:
        form = GradientBoostingForm()
    return render(request, 'gradient_boosting.html', {'form': form})



from django.shortcuts import render
from django.http import HttpResponseBadRequest
from .form import SVMForm
from .utils import train_svm

def svm_view(request):
    if request.method == 'POST':
        # Check if the form contains both dataset and target column name
        form = SVMForm(request.POST, request.FILES)
        if form.is_valid():
            # Extract the target column name from the form
            target_column_name = form.cleaned_data['target_column_name']

            # Process the uploaded dataset
            dataset_file = form.cleaned_data['dataset_file']
            if dataset_file.name.endswith('.csv'):
                dataset = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith('.xlsx') or dataset_file.name.endswith('.xls'):
                dataset = pd.read_excel(dataset_file)
            else:
                # Handle unsupported file formats or raise an error
                return HttpResponseBadRequest("Unsupported file format")

            # Call the SVM training function
            accuracy = train_svm(dataset, target_column_name)

            # Render the results template with the accuracy
            return render(request, 'svm_results.html', {'accuracy': accuracy})
    else:
        form = SVMForm()
    return render(request, 'svm.html', {'form': form})



def knn_view(request):
    if request.method == 'POST':
        # Check if the form contains both dataset and target column name
        form = KNNForm(request.POST, request.FILES)
        if form.is_valid():
            # Extract the target column name from the form
            target_column_name = form.cleaned_data['target_column_name']

            # Process the uploaded dataset
            dataset_file = form.cleaned_data['dataset_file']
            if dataset_file.name.endswith('.csv'):
                dataset = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith('.xlsx') or dataset_file.name.endswith('.xls'):
                dataset = pd.read_excel(dataset_file)
            else:
                # Handle unsupported file formats or raise an error
                return HttpResponseBadRequest("Unsupported file format")

            # Call the KNN training function
            accuracy = train_knn(dataset, target_column_name)

            # Render the results template with the accuracy
            return render(request, 'knn_results.html', {'accuracy': accuracy})
    else:
        form = KNNForm()
    return render(request, 'knn.html', {'form': form})


from django.shortcuts import render
from django.http import HttpResponseBadRequest
from .form import Naive_Bayes_Form
from .utils import train_naive_bayes
from sklearn.metrics import f1_score

def naive_bayes_view(request):
    if request.method == 'POST':
        # Check if the form contains both dataset and target column name
        form = Naive_Bayes_Form(request.POST, request.FILES)
        if form.is_valid():
            # Extract the target column name from the form
            target_column_name = form.cleaned_data['target_column_name']

            # Process the uploaded dataset
            dataset_file = form.cleaned_data['dataset_file']
            if dataset_file.name.endswith('.csv'):
                dataset = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith('.xlsx') or dataset_file.name.endswith('.xls'):
                dataset = pd.read_excel(dataset_file)
            else:
                # Handle unsupported file formats or raise an error
                return HttpResponseBadRequest("Unsupported file format")

            # Call the Naive Bayes training function
            accuracy = train_naive_bayes(dataset, target_column_name)

            # Render the results template with the accuracy
            return render(request, 'naive_bayes_results.html', {'accuracy': accuracy})
    else:
        form = Naive_Bayes_Form()
    return render(request, 'naive_bayes.html', {'form': form})


from django.shortcuts import render
from django.http import HttpResponseBadRequest
from .form import TargetColumnForm
from .utils import train_kmeans

from django.shortcuts import render
from django.http import HttpResponseBadRequest
from .form import Kmeans_clustering_Form
from .utils import train_kmeans

def kmeans_view(request):
    if request.method == 'POST':
        # Check if the form contains the dataset file
        form = Kmeans_clustering_Form(request.POST, request.FILES)
        if form.is_valid():
            # Process the uploaded dataset
            dataset_file = form.cleaned_data['dataset_file']
            if dataset_file.name.endswith('.csv'):
                dataset = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith('.xlsx') or dataset_file.name.endswith('.xls'):
                dataset = pd.read_excel(dataset_file)
            else:
                # Handle unsupported file formats or raise an error
                return HttpResponseBadRequest("Unsupported file format")

            # Get the number of clusters from the form
            num_clusters = form.cleaned_data['num_clusters']

            # Train the K-Means clustering model
            clusters = train_kmeans(dataset, num_clusters=num_clusters)

            # Render the results template with the clusters
            return render(request, 'kmeans_results.html', {'clusters': clusters})
    else:
        form = Kmeans_clustering_Form()
    return render(request, 'kmeans.html', {'form': form})


from django.shortcuts import render
from .form import hierarchical_clustering_Form
from .utils import train_hierarchical_clustering

def hierarchical_clustering_view(request):
    if request.method == 'POST':
        # Check if the form contains the dataset file
        form = hierarchical_clustering_Form(request.POST, request.FILES)
        if form.is_valid():
            # Process the uploaded dataset
            dataset_file = form.cleaned_data['dataset_file']
            if dataset_file.name.endswith('.csv'):
                dataset = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith('.xlsx') or dataset_file.name.endswith('.xls'):
                dataset = pd.read_excel(dataset_file)
            else:
                # Handle unsupported file formats or raise an error
                return HttpResponseBadRequest("Unsupported file format")

            # Get the number of clusters from the form
            num_clusters = form.cleaned_data['num_clusters']

            # Perform hierarchical clustering
            clusters = train_hierarchical_clustering(dataset, num_clusters=num_clusters)

            # Render the results template with the clusters
            return render(request, 'hierarchical_results.html', {'clusters': clusters})
    else:
        form = hierarchical_clustering_Form()
    return render(request, 'hierarchical.html', {'form': form})
