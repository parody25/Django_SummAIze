from django.urls import path
from . import views
from . import old_views
from .views import get_crm_value
from .views import separate_embedding_upload_pdfs

urlpatterns = [
    path('upload/', old_views.upload_pdfs, name='upload_pdfs'),
    path('download/', old_views.download_pdf, name='download_pdf'),
    path('chat/', old_views.chat_with_pdf, name='chat_with_pdf'),
    path('download_doc/', views.download_doc, name='download_doc'),
    path('get_crm_value/', get_crm_value, name='get_crm_value'),
    path('upload_documents/',separate_embedding_upload_pdfs, name='separate_embedding_upload_pdfs'),
    path('webscrapping/',views.web_scrapping, name='web_scrapping'),
    path('get_section/', views.get_section, name='get_section'),
    path('edit_section/', views.edit_section, name='edit_section'),
    path('collateral_qa/', views.collateral_qa, name='collateral-qa'),
    path('management_qa/', views.management_qa, name='management-qa'),
    path('management_chat/', views.management_chat, name='management-chat'),
    path('collateral_chat/', views.collateral_chat, name='collateral-chat'),
    path('get_financial_ratios/', views.get_financial_ratios, name='get_financial_ratios'),
    path('get_financial_analysis/', views.get_financial_analysis, name='get_financial_analysis'),
    path('risk_analysis/',views.risk_analysis,name='risk-analysis'),
    path('get_justification/', views.get_justification, name='get_justification'),
    path("test-redis/", views.test_redis_connection, name = 'test_redis_connection'),
    path("financial_ratio_rule_engine/", views.financial_ratio_rule_engine, name = 'financial_ratio_rule_engine')
]