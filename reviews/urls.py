
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('eda/<int:dataset_id>/', views.eda, name='eda'),
    path('train/<int:dataset_id>/', views.train, name='train'),
    path('results/<int:dataset_id>/', views.results, name='results'),
]
