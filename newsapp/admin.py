from django.contrib import admin

# Register your models here.
'''
superuser credentials
------------------------
username: mandeep
email: msdeep14.ms@gmail.com
password: pawan231
'''
from newsapp.models import Users

# Register your models here.
admin.site.register(Users)
