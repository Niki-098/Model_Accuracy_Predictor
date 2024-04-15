from django.shortcuts import render

# Create your views here.

# File: ml_app/views.py

from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Dataset, PreprocessingTechnique, MLModel, UserSelection
from .forms import DatasetUploadForm, PreprocessingForm, MLModelForm, ModelComparisonForm

def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save()
            dataset.preprocess_and_store_data()  # Preprocess and store the uploaded dataset
            messages.success(request, 'Dataset uploaded successfully!')
            return redirect('preprocessing_selection', dataset_id=dataset.id)
    else:
        form = DatasetUploadForm()
    return render(request, 'upload_dataset.html', {'form': form})

def select_preprocessing_technique(request, dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)
    if request.method == 'POST':
        form = PreprocessingForm(request.POST)
        if form.is_valid():
            preprocessing_technique = form.cleaned_data['preprocessing_technique']
            # Process the selected preprocessing technique
            # Save user selection
            messages.success(request, 'Preprocessing technique selected successfully!')
            return redirect('ml_model_selection', dataset_id=dataset.id)
    else:
        form = PreprocessingForm()
    return render(request, 'select_preprocessing_technique.html', {'form': form})

def select_ml_model(request, dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)
    if request.method == 'POST':
        form = MLModelForm(request.POST)
        if form.is_valid():
            ml_model = form.cleaned_data['ml_model']
            # Train the selected ML model
            # Save user selection
            messages.success(request, 'ML model selected successfully!')
            return redirect('compare_models', dataset_id=dataset.id)
    else:
        form = MLModelForm()
    return render(request, 'select_ml_model.html', {'form': form})

def compare_models(request, dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)
    if request.method == 'POST':
        form = ModelComparisonForm(request.POST)
        if form.is_valid():
            # Compare model accuracies and generate visualization
            messages.success(request, 'Model comparison completed successfully!')
            return redirect('result_visualization', dataset_id=dataset.id)
    else:
        form = ModelComparisonForm()
    return render(request, 'compare_models.html', {'form': form})

def result_visualization(request, dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)
    # Fetch model results and generate visualization
    return render(request, 'result_visualization.html', {'dataset': dataset})

