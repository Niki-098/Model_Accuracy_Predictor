import os
from django.conf import settings
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
        
        # Redirect to the preprocessed_files view after successful preprocessing
        return JsonResponse({'success': True, 'redirect_url': reverse('preprocessed_files')})
    else:
        # Fetch all uploaded files by the current user
        files = UploadedFile.objects.filter(uploaded_by=request.user)
        return render(request, 'preprocess.html', {'files': files})

