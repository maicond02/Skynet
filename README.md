# 🏎️ Sistema Inteligente de Detecção de Atividades Violentas com FastAPI + YOLOv8

Este projeto utiliza visão computacional com modelos YOLOv8, combinando detecção em tempo real, envio automático de alertas via Telegram e chamadas de emergência. Desenvolvido com performance turbo, aproveita lotes de frames e otimização com GPU CUDA.

---

## 🚀 Funcionalidades

- 🔍 **Detecção em Tempo Real** de atividades violentas (classe 1)
- 🧠 Inferência em lote para melhor desempenho (FRAME BUFFER)
- 📦 Suporte a múltiplos modelos (YOLOv8 personalizados)
- 📲 Envio automático de **vídeo** via Telegram
- 📞 Chamada de emergência integrada (ex: com Twilio)
- 🔔 Sistema de alerta por severidade (MILD/HIGH)
- 🧪 API via FastAPI para integrar com dashboards ou React frontend
- 🧠 Resumo de incidentes detectados
- 📹 Salvamento automático de clipes relevantes

---

## 🗂️ Estrutura do Projeto

app/
├── detection.py # Núcleo de inferência e rastreamento de severidade
├── config.py # Caminhos dos modelos YOLO
├── telegram_alert.py # Integração com Telegram Bot
├── millis_call.py # Gatilho para chamadas de emergência
main_fastapi.py # Servidor FastAPI e gerador de vídeo em tempo real

markdown
Copy
Edit

---

## 🔧 Tecnologias e Bibliotecas

- `ultralytics` - YOLOv8 para detecção de objetos
- `torch`, `numpy`, `opencv-python` - Visão computacional e tensor operations
- `fastapi`, `uvicorn` - API web moderna e assíncrona
- `python-telegram-bot` - Bot de notificação com envio de vídeo
- `twilio` - Integração com chamada de emergência
- `jinja2`, `python-dotenv`, `pydantic` - Suporte à API
- `concurrent.futures`, `threading` - Execução paralela

---

## 🔥 Como Funciona

### 📸 1. Leitura e Inferência
- A cada frame capturado do vídeo:
  - Adicionado ao buffer.
  - Quando o lote está completo, roda `MODEL1` (classe violência).
  - Resultado armazenado numa fila de inferência (`deque`).

### 🧠 2. Análise de Severidade
- Um `SeverityTracker` analisa os últimos `N` frames:
  - Se a confiança e a contagem ultrapassam os thresholds, dispara alerta.
  - MILD → HIGH se o comportamento persistir.

### 📤 3. Alertas Inteligentes
- **Telegram**: vídeo do incidente + mensagem detalhada
- **Chamada**: realizada automaticamente via `make_emergency_call` se HIGH

### 🎥 4. Visualização
- Endpoint `/video_feed` entrega stream MJPEG com os bounding boxes
- Overlay com:
  - Severidade atual
  - Confiança máxima
  - Total de detecções

---

## 🌐 Endpoints Principais

| Método | Rota                  | Descrição                                 |
|--------|-----------------------|-------------------------------------------|
| GET    | `/video_feed`         | Stream MJPEG com detecções em tempo real  |
| GET    | `/status_view`        | Status atual de severidade/confiança      |
| GET    | `/incidents`          | Histórico de incidentes detectados        |
| GET    | `/settings`           | Configurações ativas                      |
| POST   | `/update_settings`    | Atualiza diretório de vídeo e intervalos  |

---

## ⚙️ Configurações

As configurações dinâmicas podem ser alteradas via API:

```json
{
  "telegram_alert_interval": 10,
  "emergency_call_interval": 30,
  "video_save_path": "output"
}
▶️ Executando o Projeto
1. Instale as dependências:
bash
Copy
Edit
pip install -r requirements.txt
2. Crie o .env com as chaves do Telegram e Twilio:
env
Copy
Edit
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TWILIO_SID=...
TWILIO_TOKEN=...
TWILIO_PHONE=...
DEST_PHONE=...
3. Rode o servidor:
bash
Copy
Edit
uvicorn main_fastapi:app --host 0.0.0.0 --port 8001 --reload
4. Acesse:
Stream: http://localhost:8001/video_feed

Status: http://localhost:8001/status_view

📊 Exemplo de Resposta /status_view
json
Copy
Edit
{
  "level": "HIGH",
  "max_confidence": 0.92,
  "detections": 37,
  "last_update": "11:42:10",
  "alert": "High alert triggered: Telegram alert sent, emergency call initiated.",
  "logs": [...]
}
📁 Exemplo de Arquivo de Detecção Salvo
bash
Copy
Edit
output/violent_clip_1717584329.mp4
💡 Ideias Futuras
🔐 Autenticação de usuário via JWT

🗺️ Integração com mapa/monitoramento por geolocalização

🧠 Múltiplas classes de perigo (armas, multidões, fugas)

☁️ Upload automático para nuvem

📈 Dashboard em React com gráficos e timeline

🧠 Créditos
Este projeto combina IA e segurança pública com propósito social e ético, respeitando privacidade e a Constituição. Nosso objetivo é proteger vidas com inteligência.

🧪 Requisitos
Python 3.10+

Placa com suporte CUDA (ou CPU)

Modelos .pt treinados com YOLOv8

Conexão com internet (para envio de alertas)

📃 Licença
UNISAL © 2025 – Projeto desenvolvido com 💙 por Tilapia Mecânica