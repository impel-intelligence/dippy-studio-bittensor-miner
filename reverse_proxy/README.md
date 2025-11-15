# Bittensor Miner Reverse Proxy

A FastAPI-based reverse proxy server for Bittensor subnet miners with Epistula authentication.

## Overview

This reverse proxy serves as the public-facing entrypoint for the Bittensor miner, providing:
- Epistula protocol authentication 
- Request routing to internal inference server
- Security layer between external validators and internal services
- Support for both async (callback-based) and sync (direct return) inference workflows

## Setup

### Prerequisites

- Python 3.8 or higher
- A valid Bittensor hotkey for authentication
- Enough alpha stake. The alpha stake required will be dynamic and will be available at https://sn11.dippy-bittensor-subnet.com/alpha once all validators are online 
- H100 PCIe with the following specific `nvidia-smi` configuration
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
```

### Quickstart

1. Generate a working configuration (creates `reverse_proxy/config.json` if missing):
   ```bash
   make reverse-proxy-config
   ```
2. Edit the newly created config and populate at least the miner hotkey field:
   ```json
   "auth": {
     "miner_hotkey": "<your-hotkey>",
     "allowed_delta_ms": 8000,
     "cache_duration": 3600,
     "chain_endpoint": "wss://entrypoint-finney.opentensor.ai:443"
   }
   ```
   Update `services.inference_server_url` if your backend is not on `http://localhost:8091`.
3. Install the reverse proxy dependencies:
   ```bash
   make reverse-proxy-setup
   ```
4. Launch the proxy (runs FastAPI/uvicorn using the config above):
   ```bash
   make reverse-proxy-run
   ```

`reverse-proxy-run` honours the `REVERSE_PROXY_CONFIG_PATH` environment variable, so you can point to alternate configs as needed. The `reverse-proxy-dev` target is an alias for the same command.

### Public Endpoints

- `/check/{identifier}` – Returns HTTP 200 with `{ "status": "ok" }` when the path parameter matches the configured miner hotkey. This is an unauthenticated sanity check validators can use to confirm the proxy is advertising the expected identity.
- `/capacity` – Requires a valid Epistula-signed request and responds with the proxy's advertised capabilities (`{"inference": [...]}`). Validators use this to understand which inference workloads your miner can currently serve.

### Miner Registry System

The Miner Registry system allows miners to register their network endpoints (IP address and port) on the Bittensor blockchain, enabling validators to discover and connect to miners.

> **Important**: Validators on subnet 11 (netuid 11) read these commitments to discover miners.

#### Overview

The miner registry replaces the previous model submission system with a simplified approach that stores only essential networking information:
- **Address**: IP address or hostname (max 45 characters)
- **Port**: Service port number (0-65535, max 5 characters)

The total commit string is limited to 128 bytes to comply with blockchain constraints.

#### Prerequisites

- Bittensor wallet set up with coldkey and hotkey
- Access to Bittensor mainnet (netuid 11)
- Python environment with bittensor installed

#### Commands

The `register.py` script provides three main commands:

##### 1. Register Command

Register your miner's network endpoint on the blockchain.

```bash
python register.py register --network finney --netuid 11 --wallet.name <coldkey> --wallet.hotkey <hotkey> --address <ip_address> --port <port> --online
```

**Parameters:**
- `--network finney`: Use mainnet (required)
- `--netuid 11`: Subnet ID 11 (required)
- `--wallet.name`: Your coldkey name
- `--wallet.hotkey`: Your hotkey name
- `--address`: Your miner's IP address or hostname (e.g., "13.89.38.129")
- `--port`: Your miner's service port (e.g., "9999")
- `--online`: Actually commit to blockchain (omit for dry run)

**Example:**
```bash
python register.py register --network finney --netuid 11 --wallet.name coldkey --wallet.hotkey hotkey11 --address "13.89.38.129" --port "9999" --online
```

##### 2. Check Command

View all registered miners on the subnet.

```bash
python register.py check --network finney --netuid 11 [--verbose]
```

**Parameters:**
- `--network finney`: Use mainnet (required)
- `--netuid 11`: Subnet ID 11 (required)
- `--verbose`: Show detailed error information for invalid entries (optional)

**Example (basic):**
```bash
python register.py check --network finney --netuid 11
```

**Example (verbose):**
```bash
python register.py check --network finney --netuid 11 --verbose
```

##### 3. Read Command

Read all registrations (legacy format support).

```bash
python register.py read --network finney --netuid 11
```

#### Output Format

##### Successful Registration
When successfully registered, you'll see:
- Coldkey and hotkey addresses
- Network confirmation (finney)
- Compressed string that was committed
- Success message with transaction details

##### Check Command Output
The check command displays:
- Total number of commitments
- Valid commitments section showing:
  - Hotkey
  - Block number
  - Decoded address and port
- Invalid commitments section showing:
  - Hotkey
  - Raw string
  - Error details (only with --verbose)

#### Troubleshooting

##### Common Errors

1. **"Address exceeds maximum length"**
   - Ensure your address is under 45 characters

2. **"Port must be between 0 and 65535"**
   - Use a valid port number

3. **"Commit string length exceeds maximum"**
   - This shouldn't happen with the new format, but check your inputs

4. **No commitments found**
   - Verify you're using the correct network and netuid
   - Ensure your wallet has registered miners

##### Debug Mode

For detailed logging, add `--logging.debug`:

```bash
python register.py register --network finney --netuid 11 --wallet.name coldkey --wallet.hotkey hotkey11 --address "13.89.38.129" --port "9999" --logging.debug --online
```

#### Important Notes

1. **Mainnet**: The examples assume subnet 11 on mainnet (`finney`)
2. **No Hotkey Storage**: The hotkey is not stored in the registry data
3. **Simple Format**: Only address:port is stored (e.g., "13.89.38.129:9999")
4. **Size Limit**: Total string must be under 128 characters
5. **Validation**: Port must be a valid number between 0-65535

#### Migration from Old System

If you see entries with the old format (containing namespace, model names, etc.), these will appear in the "Invalid Commitments" section when using the check command. Only new registrations using the address:port format will be valid.

## Usage

### Running the Server

#### Using the CLI command (after installation)

```bash
reverse-proxy
```

#### Using Python module

```bash
python -m reverse_proxy.server
```

#### Direct execution

```bash
python reverse_proxy/server.py
```

### API Endpoints

#### Async Endpoints (Callback-based)
- `POST /inference` - Submit async inference job (requires authentication)
- `GET /inference/status/{job_id}` - Poll job status
- `POST /edit` - Submit async image editing job (requires authentication)
- `GET /edit/status/{job_id}` - Poll edit job status

#### Sync Endpoints (Direct return)
- `POST /sync/inference` - Submit sync inference job, returns image directly (requires authentication)
- `POST /sync/edit` - Submit sync edit job, returns image directly (requires authentication)

#### System Endpoints
- `GET /` - Health check endpoint
- `GET /status` - Service status with component readiness
- `GET /capacity` - Miner capacity information

### Authentication

All inference and edit endpoints require Epistula authentication. Include the appropriate authorization headers as per the Epistula protocol specification.

### Image Delivery

**Async endpoints** (`/inference`, `/edit`):
- Miner acknowledges job immediately with `job_id` and `status_url`
- Miner **always uploads** the completed image to your `callback_url` via POST
- No result/download endpoints - images are only delivered via callback

**Sync endpoints** (`/sync/inference`, `/sync/edit`):
- Image is returned directly in the HTTP response as binary data
- No callbacks or polling needed

## Development

### Code Quality Tools

The project includes comprehensive tooling configuration:

```bash
# Format code
black reverse_proxy/

# Sort imports
isort reverse_proxy/

# Lint code
flake8 reverse_proxy/

# Type checking
mypy reverse_proxy/

# Run tests
pytest
```

### Project Structure

```
reverse_proxy/
├── __init__.py              # Package initialization
├── __main__.py              # Module execution entry point
├── server.py                # Main FastAPI server
├── config.py                # Configuration management
├── env.example              # Environment variables example
├── epistula/                # Authentication module
│   ├── __init__.py
│   └── lib.py              # Epistula verifier implementation
├── pyproject.toml          # Project configuration
├── requirements.txt        # Legacy requirements (see pyproject.toml)
└── README.md               # This file
```

### Dependencies

Core dependencies:
- `bittensor` - Bittensor subnet integration
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `substrate-interface` - Substrate blockchain interface
- `loguru` - Structured logging
- `base58` - Address encoding
- `websockets` - WebSocket support
- `python-dotenv` - Environment variable loading

Development dependencies include testing, linting, and formatting tools.

## Configuration

All configuration is handled through environment variables. See the Environment Configuration section above for details.

### Security Considerations

- The reverse proxy should be the only public-facing component
- Internal inference servers should only be accessible from the reverse proxy
- Configure CORS settings appropriately for your deployment environment
- Ensure proper firewall rules are in place

## Monitoring

The server provides structured logging with configurable levels. Monitor the following:

- Authentication success/failure rates
- Request proxying performance
- Internal service connectivity
- Error rates and patterns

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify `MINER_HOTKEY` is correctly set
   - Check chain endpoint connectivity
   - Validate Epistula protocol implementation

2. **Service Connectivity**
   - Verify internal service URLs are correct
   - Check that inference servers are running
   - Validate network connectivity between services

3. **Performance Issues**
   - Monitor authentication cache hit rates
   - Check internal service response times
   - Review logging levels and output

### Logs

Logs are structured and include:
- Timestamp
- Log level
- Module/function information
- Detailed message content

Configure log level via the `LOG_LEVEL` environment variable.

## License

MIT License - see the main project repository for details. 
