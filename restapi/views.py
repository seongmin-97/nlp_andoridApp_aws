from django.shortcuts import render, redirect
from .models import Movie, movieTitle, movieInfo, sentiment
from .serializers import MovieSerializer, movieTitleSerializer, movieInfoSerializer, sentimentSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view

import os
import re
import time
import joblib
import numpy as np
import pandas as pd
from eunjeon import Mecab
from gensim.models import FastText


# Create your views here.

@api_view(['GET'])
def movie(request, title) :
    search = title.replace(' ', '')
    getMovie = movieInfo.objects.filter(titleNoSpace__contains=search)
    result = movieInfoSerializer(getMovie, many=True)
    return Response(result.data)

@api_view(['GET'])
def allMovies(request) :
    getMovie = movieTitle.objects.all()
    result = movieTitleSerializer(getMovie, many=True)
    return Response(result.data)

@api_view(['GET'])
def emotion(request, rating) :
    getSentiment = sentiment.objects.filter(rating=rating)
    result = sentimentSerializer(getSentiment, many=True)
    return Response(result.data)

@api_view(['GET'])
def movie_Info(request, title) :
    search = title.replace(' ', '')
    getMovie = movieInfo.objects.filter(titleNoSpace=search)
    result = movieInfoSerializer(getMovie, many=True)
    return Response(result.data)

def predict_(request) :
    if request.method == "GET" :
        start = time.time()
        review = request.GET["review"]
        print("Get Review _________________________________")

        pkl = joblib.load('./restapi/reviewSentiment.pkl')
        model = FastText.load('./restapi/FastText_embedding.model').wv
        print("load Model __________________________________")

        okt = Mecab()
        review_text = re.sub
        review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", review)
        word_review = okt.morphs(review_text)
        word_review = ' '.join(word_review)
        print("preprocessing _______________________")

        feature_vector = np.zeros((100), dtype=np.float32)
        num_words = 0
        index2word_set = set(model.wv.index2word)
        for w in word_review.split() :
            if w in index2word_set :
                num_words += 1
                feature_vector = np.add(feature_vector, model.wv[w])
        feature_vector = np.divide(feature_vector, num_words)
        print("predict _____________________________________")

        result = pkl.predict([feature_vector])
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        if result[0] == 1 :
            return redirect('emotion/1')
        else:
            return redirect('emotion/0')