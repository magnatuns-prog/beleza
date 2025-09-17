from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Variáveis globais para o modelo
model = None
tokenizer = None
chat_histories = {}

def load_model():
    """Carrega o modelo apenas quando necessário"""
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.info("Carregando modelo DialoGPT-small...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        logger.info("Modelo carregado com sucesso!")
    
    return model, tokenizer

@app.route('/')
def home():
    return jsonify({
        "message": "API do Chat IA está funcionando!",
        "status": "online",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Mensagem vazia'}), 400
        
        # Carregar modelo se necessário
        model, tokenizer = load_model()
        
        # Obter ou criar histórico para esta sessão
        if session_id not in chat_histories:
            chat_histories[session_id] = None
        
        # Codificar a nova mensagem do usuário
        new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
        
        # Append ao histórico de chat, se existir
        if chat_histories[session_id] is not None:
            bot_input_ids = torch.cat([chat_histories[session_id], new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids
        
        # Gerar resposta
        chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=3
        )
        
        # Decodificar resposta
        bot_response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        
        # Atualizar histórico
        chat_histories[session_id] = chat_history_ids
        
        return jsonify({'response': bot_response})
    
    except Exception as e:
        logger.error(f"Erro no chat: {str(e)}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
