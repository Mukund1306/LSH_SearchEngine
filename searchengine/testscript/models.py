from django.db import models

class Testscript(models.Model):
    
    tweet = models.CharField(max_length=10000)
    
    
    def __str__(self):
        return self.name
# Create your models here.
