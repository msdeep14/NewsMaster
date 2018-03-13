from django.shortcuts import render

# Create your views here.



def train_model():
    perform_preprocessing()








def newsfeed(request):
    print("REMOVE THIS ONCE DEBUGGING IS DONE!!!\n")


    return render(request, 'newsapp/newsfeed.html', {'url_list': {}})
