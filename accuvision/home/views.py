import os
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from .form import UploadForm


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
            # Render the success page with the uploaded file name
            return render(request, 'upload_success.html', {'filename': uploaded_file.name})
    else:
        form = UploadForm()
    return render(request, 'upload_form.html', {'form': form})


def upload_success(request, filename):
    # Pass uploaded file information to the template
    context = {'filename': filename}
    return render(request, 'upload_success.html', context)
