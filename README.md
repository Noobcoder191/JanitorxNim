# Lorebary DeepSeek 3.2 Proxy

A proxy server that connects Lorebary (or any OpenAI-compatible client) to NVIDIA NIM's DeepSeek 3.2 model. Features a web-based control panel for easy configuration.

## Features

- 🔄 Proxy compatibility - works with Lorebary and other proxies
- 🎛️ Web-based control panel for runtime configuration
- 🌊 Streaming support for real-time responses
- 🧠 Optional thinking/reasoning mode
- 📊 Request logging and monitoring

## Prerequisites

- [Node.js](https://nodejs.org/) (v24 LTS)
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) (for exposing localhost)
- NVIDIA NIM API key (get one from [NVIDIA](https://build.nvidia.com/))
- Dotenv (node has to already be installed before installing dotenv)
  
## Installation

1. Download the files or clone the repo

2. Install dependencies (the cmd has to be in the same directory as the other files):
```bash
npm install
```

3. Create a `.env` file in the project root:
```env
NIM_API_KEY=your_nvidia_api_key_here
```

4. Move the cloudflared executable to the root folder, then rename it to "cloudflared.exe".

5. Install dotenv by running "npm i dotenv" in your command prommpt. 

## Usage

### Windows

1. **Start the proxy**:
   - Simply double-click `start-lorebary-proxy.bat`
   - A Cloudflare tunnel URL will appear in one of the cmd windows (e.g., `https://something.trycloudflare.com`)
   - Copy this URL to use in the proxy

2. **If you're using LoreBary**:
   - Open Lorebary custom proxy settings
   - Set target url to your Cloudflare URL (YOU NEED TO ADD /v1/chat/completions IN THE END OF THE URL)
   - Copy your custom lorebary url to janitor, put sommething random in the api key field, and put deepseek-reasoner in the model name

### Linux/Mac

1. **Start the proxy**:
```bash
npm start
```

2. **In a separate terminal, start Cloudflare tunnel**:
```bash
cloudflared tunnel --url http://localhost:3000
```

3. Copy the generated URL and use it in Lorebary

 (I HAVEN'T TESTED IT IN LINUX/MAC YET)

## Control Panel

Access the web control panel at `http://localhost:3001` to configure:

- **Show Reasoning**: Display model's thinking process in `<think>` tags
- **Enable Thinking Mode**: Send thinking parameter to the model
- **Request Logging**: Log all requests to console
- **Streaming**: Enable/disable streaming responses
- **Max Tokens**: Set maximum response length (1-8192)
- **Temperature**: Adjust creativity level (0-1)

## Configuration

The proxy supports runtime configuration through the control panel. Settings are stored in memory and reset on restart.

Default settings:
```javascript
{
  showReasoning: false,
  enableThinking: false,
  logRequests: true,
  maxTokens: 4096,
  temperature: 0.7,
  streamingEnabled: true
}
```

## Troubleshooting

### "NIM_API_KEY environment variable not set"
- Make sure you created a `.env` file with your NVIDIA API key
- Restart the server after creating the `.env` file

### Port already in use
- The proxy uses ports 3000 and 3001
- Close any applications using these ports or edit `server.js` to use different ports

### Cloudflare tunnel not starting
- Make sure `cloudflared` is installed and in your PATH
- Download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software for any purpose, commercial or non-commercial, and by any means.

## Support

If you encounter issues:
1. Check the console logs for error messages
2. Verify your API key is correct
3. Ensure all dependencies are installed
4. Open an issue on GitHub with details

This is very WIP, i plan on adding a bunch more features.

95%~ made with AI.
