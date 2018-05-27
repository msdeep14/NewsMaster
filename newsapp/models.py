from __future__ import unicode_literals
from django.db import models

# Create your models here.

from django.contrib.auth.models import User

from django.utils import timezone

class Users(models.Model):
	user = models.ForeignKey(User,)
	time_added = models.DateTimeField(default=timezone.now)
	categories = models.CharField(max_length = 100, default = 'top')

	def __str__(self):
		return self.user.username
