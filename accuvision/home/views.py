import os
from django.conf import settings
from django.forms import ModelForm
from django.http import HttpResponse
from django.shortcuts import render, redirect
import pandas as pd
from .form import UploadForm, UploadFileForm
from .models import UploadedFile
from django.http import JsonResponse



# Create your views here.
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
from .utils import auto_preprocess_dataset

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

