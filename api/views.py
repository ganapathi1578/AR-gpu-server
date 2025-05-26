from django.shortcuts import render
from django.http import JsonResponse
import tempfile
from utils import preprocess_video, model_predict_lable_101
import os

# Create your views here.



def iwantvid(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        if not video_file:
            return JsonResponse({'error': 'No video uploaded'}, status=400)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            for chunk in video_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name  # full path to the saved video
        
        filename = video_file.name  # get original filename
        save_path = os.path.join(os.getcwd(), filename)  # current working directory + filename


        with open(save_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)

        # Process the video to a tensor
        video_tensor = preprocess_video(tmp_path, 64, 1, 112)
        label = model_predict_lable_101(video_tensor)

        return JsonResponse({'label': label}, status=200)

    return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    



def pred(request):
    if request.method == 'GET':
        return render(request, 'predict.html')  # Serve HTML

    elif request.method == 'POST':
        video_file = request.FILES.get('video')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
            for chunk in video_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name

        try:
            video_tensor = preprocess_video(tmp_path, 64, 1, 112)
            label = model_predict_lable_101(video_tensor)
        finally:
            os.remove(tmp_path)

        return JsonResponse({'label': label})
