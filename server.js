require('dotenv').config();
// server.js - Lorebary-optimized NVIDIA NIM Proxy for DeepSeek V3.2
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;
const CONTROL_PANEL_PORT = 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased for large Lorebary commands

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Runtime configuration (can be changed without restart)
let config = {
  showReasoning: false,
  enableThinking: false,
  logRequests: true,
  maxTokens: 4096,
  temperature: 0.7,
  streamingEnabled: true
};

// Lorebary-optimized model mapping
const MODEL_MAPPING = {
  'gpt-4o': 'deepseek-ai/deepseek-v3.2',
  'gpt-4': 'deepseek-ai/deepseek-v3.2',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.2',
  'deepseek-chat': 'deepseek-ai/deepseek-v3.2',
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3.2'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'Lorebary DeepSeek V3.2 NVIDIA NIM Proxy',
    model: 'deepseek-ai/deepseek-v3.2',
    api_key_configured: !!NIM_API_KEY,
    config: config
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Lorebary DeepSeek V3.2 Proxy',
    endpoints: {
      health: '/health',
      models: '/v1/models',
      chat: '/v1/chat/completions'
    }
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim',
    permission: [],
    root: model,
    parent: null
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'NIM_API_KEY environment variable not set',
          type: 'configuration_error',
          code: 500
        }
      });
    }

    const { model, messages, temperature, max_tokens, stream, top_p, frequency_penalty, presence_penalty } = req.body;
    
    // Get NVIDIA model (default to DeepSeek V3.2)
    const nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v3.2';
    
    if (config.logRequests) {
      console.log(`[Lorebary] Request: ${model} -> ${nimModel}`);
      console.log(`[Lorebary] Messages: ${messages.length} messages, Stream: ${stream || config.streamingEnabled}`);
    }
    
    // Build NIM request - preserve ALL messages including Lorebary commands
    const nimRequest = {
      model: nimModel,
      messages: messages, // Forward everything from Lorebary
      temperature: temperature !== undefined ? temperature : config.temperature,
      max_tokens: max_tokens || config.maxTokens,
      stream: stream !== undefined ? stream : config.streamingEnabled
    };

    // Add optional parameters if provided
    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;
    
    // Add thinking mode if enabled
    if (config.enableThinking) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 120000 // 2 minute timeout for long RP responses
    });
    
    if (stream) {
      // Handle streaming response (optimized for Lorebary)
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              
              // Handle reasoning content based on config
              if (data.choices?.[0]?.delta?.reasoning_content) {
                if (config.showReasoning) {
                  // Keep reasoning content, wrap in tags
                  const reasoning = data.choices[0].delta.reasoning_content;
                  if (!data.choices[0].delta.content) {
                    data.choices[0].delta.content = `<think>${reasoning}</think>`;
                  } else {
                    data.choices[0].delta.content = `<think>${reasoning}</think>\n\n${data.choices[0].delta.content}`;
                  }
                }
                delete data.choices[0].delta.reasoning_content;
              }
              
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              // Pass through malformed lines
              res.write(line + '\n\n');
            }
          }
        });
      });
      
      response.data.on('end', () => {
        if (config.logRequests) {
          console.log('[Lorebary] Stream completed');
        }
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('[Lorebary] Stream error:', err.message);
        res.end();
      });
    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let content = choice.message.content || '';
          
          // Handle reasoning based on config
          if (config.showReasoning && choice.message?.reasoning_content) {
            content = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${content}`;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: content
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      if (config.logRequests) {
        console.log('[Lorebary] Response completed');
      }
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('[Lorebary] Proxy error:', error.response?.data || error.message);
    
    const errorMessage = error.response?.data?.error?.message || error.message || 'Internal server error';
    const statusCode = error.response?.status || 500;
    
    res.status(statusCode).json({
      error: {
        message: errorMessage,
        type: 'api_error',
        code: statusCode
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found. Use /v1/chat/completions for chat.`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Lorebary DeepSeek V3.2 Proxy running on port ${PORT}`);
  console.log(`📡 Health check: http://localhost:${PORT}/health`);
  console.log(`🤖 Model: deepseek-ai/deepseek-v3.2`);
  console.log(`🔑 API Key: ${NIM_API_KEY ? 'Configured ✓' : 'NOT SET ✗'}`);
  console.log(`🎛️  Control Panel: http://localhost:${CONTROL_PANEL_PORT}`);
});

// ============================================
// CONTROL PANEL WEB INTERFACE
// ============================================

const controlApp = express();
controlApp.use(cors());
controlApp.use(express.json());

// Serve control panel HTML
controlApp.get('/', (req, res) => {
  res.send(`
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lorebary Proxy Control Panel</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }
    
    .container {
      background: white;
      border-radius: 20px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      padding: 40px;
      max-width: 600px;
      width: 100%;
    }
    
    h1 {
      color: #333;
      margin-bottom: 10px;
      font-size: 28px;
    }
    
    .subtitle {
      color: #666;
      margin-bottom: 30px;
      font-size: 14px;
    }
    
    .status {
      background: #f0f9ff;
      border-left: 4px solid #0ea5e9;
      padding: 15px;
      border-radius: 8px;
      margin-bottom: 30px;
    }
    
    .status-item {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 14px;
    }
    
    .status-label {
      color: #666;
      font-weight: 500;
    }
    
    .status-value {
      color: #0ea5e9;
      font-weight: 600;
    }
    
    .setting {
      background: #f8fafc;
      padding: 20px;
      border-radius: 12px;
      margin-bottom: 15px;
      transition: all 0.3s;
    }
    
    .setting:hover {
      background: #f1f5f9;
      transform: translateY(-2px);
    }
    
    .setting-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    
    .setting-title {
      font-weight: 600;
      color: #333;
      font-size: 16px;
    }
    
    .setting-desc {
      color: #666;
      font-size: 13px;
      line-height: 1.5;
    }
    
    .toggle {
      position: relative;
      width: 60px;
      height: 30px;
    }
    
    .toggle input {
      opacity: 0;
      width: 0;
      height: 0;
    }
    
    .slider {
      position: absolute;
      cursor: pointer;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: #cbd5e1;
      transition: .4s;
      border-radius: 30px;
    }
    
    .slider:before {
      position: absolute;
      content: "";
      height: 22px;
      width: 22px;
      left: 4px;
      bottom: 4px;
      background-color: white;
      transition: .4s;
      border-radius: 50%;
    }
    
    input:checked + .slider {
      background-color: #10b981;
    }
    
    input:checked + .slider:before {
      transform: translateX(30px);
    }
    
    .input-group {
      background: white;
      padding: 15px;
      border-radius: 8px;
      margin-top: 10px;
    }
    
    .input-label {
      display: block;
      color: #666;
      font-size: 13px;
      margin-bottom: 8px;
      font-weight: 500;
    }
    
    input[type="number"], input[type="range"] {
      width: 100%;
      padding: 10px;
      border: 2px solid #e2e8f0;
      border-radius: 8px;
      font-size: 14px;
      transition: border-color 0.3s;
    }
    
    input[type="number"]:focus {
      outline: none;
      border-color: #667eea;
    }
    
    input[type="range"] {
      height: 6px;
      padding: 0;
    }
    
    .range-value {
      text-align: center;
      color: #667eea;
      font-weight: 600;
      margin-top: 5px;
      font-size: 14px;
    }
    
    .save-btn {
      width: 100%;
      padding: 15px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 16px;
      font-weight: 600;
      cursor: pointer;
      margin-top: 20px;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .save-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    .save-btn:active {
      transform: translateY(0);
    }
    
    .toast {
      position: fixed;
      top: 20px;
      right: 20px;
      background: #10b981;
      color: white;
      padding: 15px 25px;
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
      display: none;
      animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
      from {
        transform: translateX(400px);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }
    
    .toast.show {
      display: block;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎛️ Lorebary Proxy Control Panel</h1>
    <p class="subtitle">Configure your DeepSeek V3.2 proxy settings in real-time</p>
    
    <div class="status">
      <div class="status-item">
        <span class="status-label">Proxy Status:</span>
        <span class="status-value" id="proxyStatus">Loading...</span>
      </div>
      <div class="status-item">
        <span class="status-label">Model:</span>
        <span class="status-value">deepseek-ai/deepseek-v3.2</span>
      </div>
      <div class="status-item">
        <span class="status-label">API Key:</span>
        <span class="status-value" id="apiKeyStatus">Loading...</span>
      </div>
    </div>
    
    <div class="setting">
      <div class="setting-header">
        <div>
          <div class="setting-title">Show Reasoning</div>
          <div class="setting-desc">Display model's thinking process in &lt;think&gt; tags</div>
        </div>
        <label class="toggle">
          <input type="checkbox" id="showReasoning">
          <span class="slider"></span>
        </label>
      </div>
    </div>
    
    <div class="setting">
      <div class="setting-header">
        <div>
          <div class="setting-title">Enable Thinking Mode</div>
          <div class="setting-desc">Send thinking parameter to models that support it</div>
        </div>
        <label class="toggle">
          <input type="checkbox" id="enableThinking">
          <span class="slider"></span>
        </label>
      </div>
    </div>
    
    <div class="setting">
      <div class="setting-header">
        <div>
          <div class="setting-title">Request Logging</div>
          <div class="setting-desc">Log all requests to console for debugging</div>
        </div>
        <label class="toggle">
          <input type="checkbox" id="logRequests">
          <span class="slider"></span>
        </label>
      </div>
    </div>
    
    <div class="setting">
      <div class="setting-header">
        <div>
          <div class="setting-title">Streaming</div>
          <div class="setting-desc">Enable streaming responses (default for Lorebary)</div>
        </div>
        <label class="toggle">
          <input type="checkbox" id="streamingEnabled">
          <span class="slider"></span>
        </label>
      </div>
    </div>
    
    <div class="setting">
      <div class="setting-title">Max Tokens</div>
      <div class="setting-desc">Maximum response length (1-8192 tokens)</div>
      <div class="input-group">
        <label class="input-label">Tokens:</label>
        <input type="number" id="maxTokens" min="1" max="8192" step="1">
      </div>
    </div>
    
    <div class="setting">
      <div class="setting-title">Temperature</div>
      <div class="setting-desc">Creativity level (0 = focused, 1 = creative)</div>
      <div class="input-group">
        <input type="range" id="temperature" min="0" max="1" step="0.1">
        <div class="range-value" id="tempValue">0.7</div>
      </div>
    </div>
    
    <button class="save-btn" onclick="saveConfig()">💾 Save Configuration</button>
  </div>
  
  <div class="toast" id="toast">✓ Configuration saved successfully!</div>
  
  <script>
    // Load current configuration
    async function loadConfig() {
      try {
        const response = await fetch('http://localhost:3000/health');
        const data = await response.json();
        
        document.getElementById('proxyStatus').textContent = data.status === 'ok' ? '✓ Running' : '✗ Error';
        document.getElementById('apiKeyStatus').textContent = data.api_key_configured ? '✓ Configured' : '✗ Missing';
        
        if (data.config) {
          document.getElementById('showReasoning').checked = data.config.showReasoning;
          document.getElementById('enableThinking').checked = data.config.enableThinking;
          document.getElementById('logRequests').checked = data.config.logRequests;
          document.getElementById('streamingEnabled').checked = data.config.streamingEnabled;
          document.getElementById('maxTokens').value = data.config.maxTokens;
          document.getElementById('temperature').value = data.config.temperature;
          document.getElementById('tempValue').textContent = data.config.temperature;
        }
      } catch (error) {
        console.error('Failed to load config:', error);
        document.getElementById('proxyStatus').textContent = '✗ Not Running';
      }
    }
    
    // Update temperature display
    document.getElementById('temperature').addEventListener('input', (e) => {
      document.getElementById('tempValue').textContent = e.target.value;
    });
    
    // Save configuration
    async function saveConfig() {
      const config = {
        showReasoning: document.getElementById('showReasoning').checked,
        enableThinking: document.getElementById('enableThinking').checked,
        logRequests: document.getElementById('logRequests').checked,
        streamingEnabled: document.getElementById('streamingEnabled').checked,
        maxTokens: parseInt(document.getElementById('maxTokens').value),
        temperature: parseFloat(document.getElementById('temperature').value)
      };
      
      try {
        const response = await fetch('http://localhost:3001/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        });
        
        if (response.ok) {
          const toast = document.getElementById('toast');
          toast.classList.add('show');
          setTimeout(() => toast.classList.remove('show'), 3000);
        }
      } catch (error) {
        alert('Failed to save configuration: ' + error.message);
      }
    }
    
    // Load config on page load
    loadConfig();
    
    // Refresh status every 10 seconds
    setInterval(loadConfig, 10000);
  </script>
</body>
</html>
  `);
});

require('dotenv').config();

// Get current config
controlApp.get('/config', (req, res) => {
  res.json(config);
});

// Update config
controlApp.post('/config', (req, res) => {
  const newConfig = req.body;
  
  // Validate and update config
  if (typeof newConfig.showReasoning === 'boolean') config.showReasoning = newConfig.showReasoning;
  if (typeof newConfig.enableThinking === 'boolean') config.enableThinking = newConfig.enableThinking;
  if (typeof newConfig.logRequests === 'boolean') config.logRequests = newConfig.logRequests;
  if (typeof newConfig.streamingEnabled === 'boolean') config.streamingEnabled = newConfig.streamingEnabled;
  if (typeof newConfig.maxTokens === 'number' && newConfig.maxTokens > 0) config.maxTokens = newConfig.maxTokens;
  if (typeof newConfig.temperature === 'number' && newConfig.temperature >= 0 && newConfig.temperature <= 1) {
    config.temperature = newConfig.temperature;
  }
  
  console.log('[Control Panel] Configuration updated:', config);
  res.json({ success: true, config: config });
});

controlApp.listen(CONTROL_PANEL_PORT, () => {
  console.log(`🎛️  Control Panel available at: http://localhost:${CONTROL_PANEL_PORT}`);
});