# Create your views here.
from django.views.generic import TemplateView, ListView

import time

from . import ldatm

from .models import Testscript

class HomePageView(TemplateView):
    template_name = 'home.html'

class SearchResultsView(ListView):
    model = Testscript
    template_name = 'search_results.html'
    def get_queryset(self): 
        def rep(temp):
            f = open("C:\\Users\\Mukund\\Desktop\\dataset\\tweets.txt")
            f2 = open("C:\\Users\\Mukund\\Desktop\\dataset\\senti.txt")
            for i in range(1,16108):
                sent = f2.readline()
                sent1 = sent.split(',')
                #print(i)
                sent2 = sent1[11]
                line = f.readline()
                if(temp == i):
                        print(sent1)
                        print(sent2)
                        print(line)
                        object_list.append(sent2)
                        object_list.append(line)
            f.close()
            f2.close()

        def operation(query):
            ans = ldatm.main(query)
            return ans

        try:
        #start = time.time()
            query = self.request.GET.get('q')
            object_list = ldatm.main(query)
            #print(object_list)
            temp = object_list
            temp1 = temp[0]
            temp2 = temp[1]
            temp3 = temp[2]
            temp4 = temp[3]
            temp5 = temp[4]
            temp6 = temp[5]
            temp7 = temp[6]
            temp8 = temp[7]
            temp9 = temp[8]
            temp10 = temp[9]

            object_list = []
            rep(temp1)
            rep(temp2)
            rep(temp3)
            rep(temp4)
            rep(temp5)
            rep(temp6)
            rep(temp7)
            rep(temp8)
            rep(temp9)
            rep(temp10)
            #end = time.time()
            #print(end-start)
            return object_list
        except:
            object_list = []
            return object_list
        



    