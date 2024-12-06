from flask import Flask, request, render_template, jsonify
from .similarity_search import SearchService

app = Flask(__name__)
search_service = SearchService()

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.json['query']
        result = search_service.perform_search(query)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
