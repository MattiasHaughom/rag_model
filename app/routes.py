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
        response = search_service.perform_search(query)
        
        # Format the response into HTML
        formatted_response = {
            'key_points': response.key_points,
            'sections': response.sections,
            'sources': response.sources,
            'enough_context': response.enough_context
        }
        
        return jsonify({'success': True, 'result': formatted_response})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500