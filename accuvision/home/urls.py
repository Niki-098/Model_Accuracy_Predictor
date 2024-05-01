"""
URL configuration for accuvision project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from home import views
from django.conf import settings
from django.conf.urls.static import static
from . import views
from django.views.generic import TemplateView

urlpatterns = [
   # path('admin/', admin.site.urls),
    path('', views.home, name = 'home'),
    path('Login/index.html', views.home, name='home'),
    path('SignUp/',views.SignUp, name = 'SignUp'),
    path('Login/',views.Login, name = 'Login'),
    path('about/',views.about, name = 'about'),
    path('upload/', views.upload_file, name='upload_file'),
    path('upload_success/<str:filename>/', views.upload_success, name='upload_success'),  # URL for upload success
    path('show_uploaded_files/',views.show_uploaded_files,name='show_uploaded_files'),
    path('upload/', views.upload_files, name='upload_files'),  # URL for uploading files
    path('preprocess_files/', views.preprocess_files, name='preprocess_files'),  # URL for preprocessing
   # path('preprocessed_files/', views.preprocessed_files, name='preprocessed_files'),  # URL for preprocessing

    path('preprocessing_files/', views.preprocessing_files, name='preprocessing_files'),
    path('preprocessed_files/',TemplateView.as_view(template_name='preprocessed_files.html'), name = 'preprocessed_files'),
    path('download-preprocessed-data/', views.download_preprocessed_data, name='download_preprocessed_data'),
    path('models/', views.model_page, name='model_page'),
    path('model-selection/', views.model_selection, name='model_selection'),
    path('select_model/', views.model_select_view, name='model_select'),
    path('model_select_view/', views.model_select_view, name='model_select_view'),
    path('logistic_regression_view/', views.logistic_regression_view, name='logistic_regression_view'),
    path('decision_tree_view/', views.decision_tree_view, name='decision_tree_view'),
    path('random_forest_view/', views.random_forest_view, name='random_forest_view'),
    path('gradient_boosting_view/', views.gradient_boosting_view, name='gradient_boosting_view'),
    path('svm_view/', views.svm_view, name='svm_view'),
    path('knn_view/', views.knn_view, name='knn_view'),
    path('naive_bayes_view/', views.naive_bayes_view, name='naive_bayes_view'),
    path('kmeans_view/', views.kmeans_view, name='kmeans_view'),
    path('hierarchical_clustering_view/', views.hierarchical_clustering_view, name='hierarchical_clustering_view'),
    ] + static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)
