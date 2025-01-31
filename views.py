from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

@method_decorator(csrf_exempt, name='dispatch')
class SummarizeAPIView(View):
    def get(self, request, *args, **kwargs):
        # Handle GET request if needed
        return JsonResponse({"message": "GET request received. Use POST for summarization."})

    def post(self, request, *args, **kwargs):
        # Get the article_text from the POST request
        text = request.POST.get('text', '')

        # Tokenize sentences
        sentences = sent_tokenize(text)

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Calculate sentence scores (sum of TF-IDF weights for each sentence)
        sentence_scores = tfidf_matrix.sum(axis=1)

        # Rank sentences based on scores
        ranked_sentences = [(score, sentence) for score, sentence in zip(sentence_scores, sentences)]
        ranked_sentences.sort(reverse=True)

        # Select the top N sentences as the summary
        num_sentences_in_summary = 3
        summary = ' '.join([sentence for score, sentence in ranked_sentences[:num_sentences_in_summary]])

        # Return the summary and the original article text as a JSON response
        response_data = {
            'original': text,
            'summarized': summary,
        }

        return JsonResponse(response_data)
