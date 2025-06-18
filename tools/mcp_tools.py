import json
import os
import platform
import shutil
import subprocess

ALL_MCP_SERVERS = [
    {
        "name": "AlterX",
        "key": "AlterX",
        "command": "npx",
        "args": ["-y", "gc-alterx-mcp"],
        "description": "MCP server for subdomain permutation and wordlist generation using the AlterX tool.",
        "exe_name": "alterx.exe",
        "env_var": "ALTERX_PATH",
        "homepage": "https://www.npmjs.com/package/gc-alterx-mcp",
    },
    {
        "name": "Amass",
        "key": "Amass",
        "command": "npx",
        "args": ["-y", "gc-amass-mcp"],
        "description": "MCP server for advanced subdomain enumeration and reconnaissance using the Amass tool.",
        "exe_name": "amass.exe",
        "env_var": "AMASS_PATH",
        "homepage": "https://www.npmjs.com/package/gc-amass-mcp",
    },
    {
        "name": "Arjun",
        "key": "Arjun",
        "command": "npx",
        "args": ["-y", "gc-arjun-mcp"],
        "description": "MCP server for discovering hidden HTTP parameters using the Arjun tool.",
        "exe_name": "arjun",
        "env_var": "ARJUN_PATH",
        "homepage": "https://www.npmjs.com/package/gc-arjun-mcp",
    },
    {
        "name": "Assetfinder",
        "key": "Assetfinder",
        "command": "npx",
        "args": ["-y", "gc-assetfinder-mcp"],
        "description": "MCP server for passive subdomain discovery using the Assetfinder tool.",
        "exe_name": "assetfinder.exe",
        "env_var": "ASSETFINDER_PATH",
        "homepage": "https://www.npmjs.com/package/gc-assetfinder-mcp",
    },
    {
        "name": "Certificate Transparency",
        "key": "CrtSh",
        "command": "npx",
        "args": ["-y", "gc-crtsh-mcp"],
        "description": "MCP server for subdomain discovery using SSL certificate transparency logs (crt.sh).",
        "exe_name": None,  # No executable needed for this service
        "env_var": None,
        "homepage": "https://www.npmjs.com/package/gc-crtsh-mcp",
    },
    {
        "name": "FFUF Fuzzer",
        "key": "FFUF",
        "command": "npx",
        "args": ["-y", "gc-ffuf-mcp"],
        "description": "MCP server for web fuzzing operations using FFUF (Fuzz Faster U Fool) tool.",
        "exe_name": "ffuf.exe",
        "env_var": "FFUF_PATH",
        "homepage": "https://www.npmjs.com/package/gc-ffuf-mcp",
    },
    {
        "name": "httpx",
        "key": "HTTPx",
        "command": "npx",
        "args": ["-y", "gc-httpx-mcp"],
        "description": "MCP server for fast HTTP toolkit and port scanning using the httpx tool.",
        "exe_name": "httpx.exe",
        "env_var": "HTTPX_PATH",
        "homepage": "https://www.npmjs.com/package/gc-httpx-mcp",
    },
    {
        "name": "Hydra",
        "key": "Hydra",
        "command": "npx",
        "args": ["-y", "gc-hydra-mcp"],
        "description": "MCP server for password brute-force attacks and credential testing using the Hydra tool.",
        "exe_name": "hydra.exe",
        "env_var": "HYDRA_PATH",
        "homepage": "https://www.npmjs.com/package/gc-hydra-mcp",
    },
    {
        "name": "Katana",
        "key": "Katana",
        "command": "npx",
        "args": ["-y", "gc-katana-mcp"],
        "description": "MCP server for fast web crawling with JavaScript parsing using the Katana tool.",
        "exe_name": "katana.exe",
        "env_var": "KATANA_PATH",
        "homepage": "https://www.npmjs.com/package/gc-katana-mcp",
    },
    {
        "name": "Masscan",
        "key": "Masscan",
        "command": "npx",
        "args": ["-y", "gc-masscan-mcp"],
        "description": "MCP server for high-speed network port scanning with the Masscan tool.",
        "exe_name": "masscan.exe",
        "env_var": "MASSCAN_PATH",
        "homepage": "https://www.npmjs.com/package/gc-masscan-mcp",
    },
    {
        "name": "Nmap Scanner",
        "key": "Nmap",
        "command": "npx",
        "args": ["-y", "gc-nmap-mcp"],
        "description": "MCP server for interacting with Nmap network scanner to discover hosts and services on a network.",
        "exe_name": "nmap.exe",
        "env_var": "NMAP_PATH",
        "homepage": "https://www.npmjs.com/package/gc-nmap-mcp",
    },
    {
        "name": "Nuclei Scanner",
        "key": "Nuclei",
        "command": "npx",
        "args": ["-y", "gc-nuclei-mcp"],
        "description": "MCP server for vulnerability scanning using Nuclei's template-based detection engine.",
        "exe_name": "nuclei.exe",
        "env_var": "NUCLEI_PATH",
        "homepage": "https://www.npmjs.com/package/gc-nuclei-mcp",
    },
    {
        "name": "Scout Suite",
        "key": "ScoutSuite",
        "command": "npx",
        "args": ["-y", "gc-scoutsuite-mcp"],
        "description": "MCP server for cloud security auditing using the Scout Suite tool.",
        "exe_name": "scout.py",
        "env_var": "SCOUTSUITE_PATH",
        "homepage": "https://www.npmjs.com/package/gc-scoutsuite-mcp",
    },
    {
        "name": "shuffledns",
        "key": "ShuffleDNS",
        "command": "npx",
        "args": ["-y", "gc-shuffledns-mcp"],
        "description": "MCP server for high-speed DNS brute-forcing and resolution using the shuffledns tool.",
        "exe_name": "shuffledns",
        "env_var": "SHUFFLEDNS_PATH",
        "env_extra": {"MASSDNS_PATH": ""},
        "homepage": "https://www.npmjs.com/package/gc-shuffledns-mcp",
    },
    {
        "name": "SQLMap",
        "key": "SQLMap",
        "command": "npx",
        "args": ["-y", "gc-sqlmap-mcp"],
        "description": "MCP server for conducting automated SQL injection detection and exploitation using SQLMap.",
        "exe_name": "sqlmap.py",
        "env_var": "SQLMAP_PATH",
        "homepage": "https://www.npmjs.com/package/gc-sqlmap-mcp",
    },
    {
        "name": "SSL Scanner",
        "key": "SSLScan",
        "command": "npx",
        "args": ["-y", "gc-sslscan-mcp"],
        "description": "MCP server for analyzing SSL/TLS configurations and identifying security issues.",
        "exe_name": "sslscan.exe",
        "env_var": "SSLSCAN_PATH",
        "homepage": "https://www.npmjs.com/package/gc-sslscan-mcp",
    },
    {
        "name": "Wayback URLs",
        "key": "WaybackURLs",
        "command": "npx",
        "args": ["-y", "gc-waybackurls-mcp"],
        "description": "MCP server for discovering historical URLs from the Wayback Machine archive.",
        "exe_name": "waybackurls.exe",
        "env_var": "WAYBACKURLS_PATH",
        "homepage": "https://www.npmjs.com/package/gc-waybackurls-mcp",
    },
    {
        "name": "fetch",
        "key": "fetch",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
    },
    {
        "name": "Sequential Thinking",
        "key": "sequential-thinking",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
    },
]

scanner_keys = [
    # "AlterX",
    "Amass",
    "Arjun",
    # "Assetfinder",
    "CrtSh",
    "FFUF",
    "HTTPx",
    # "Hydra",
    # "Katana",
    # "Masscan",
    # "Nmap",
    "fetch",
    "sequential-thinking",
]

planner_keys = ["sequential-thinking"]

SCANNER_MCP_SERVERS = [srv for srv in ALL_MCP_SERVERS if srv["key"] in scanner_keys]
PLANNER_MCP_SERVERS = [srv for srv in ALL_MCP_SERVERS if srv["key"] in planner_keys]


def find_tool_path(tool_name):
    """Auto-discover tool path using system commands"""
    try:
        if platform.system() == "Windows":
            # Use 'where' command on Windows
            result = subprocess.run(
                ["where", tool_name], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                # Get first valid path from results
                paths = result.stdout.strip().split("\n")
                for path in paths:
                    path = path.strip()
                    if path and os.path.exists(path):
                        return path
        else:
            # Use 'which' command on Linux/Mac
            path = shutil.which(tool_name)
            if path and os.path.exists(path):
                return path
    except Exception:
        pass
    return None


def get_tool_search_variants(exe_name):
    """Get different variants of tool names to search for"""
    if not exe_name:
        return []

    # For Windows, just search for the base name - 'where' will find the actual executable
    base_name = exe_name.replace(".exe", "").replace(".py", "")
    variants = [base_name]

    # Also try the exact name if it's different
    if exe_name != base_name:
        variants.append(exe_name)

    return variants


def write_mcp_json_from_servers(servers_list: list[dict], json_path="mcp.json"):
    actual_path = os.path.join(os.getcwd(), json_path)
    servers = []
    for srv in servers_list:
        params = {"command": srv["command"], "args": srv["args"].copy(), "env": {}}
        # Set env var if present
        env_var = srv.get("env_var")
        exe_name = srv.get("exe_name")
        if env_var:
            if exe_name:
                variants = get_tool_search_variants(exe_name)
                tool_path = None
                for v in variants:
                    tool_path = find_tool_path(v)
                    if tool_path:
                        break
                if tool_path:
                    params["env"][env_var] = tool_path
                else:
                    value = input(
                        f"Please set the environment variable {env_var} for {srv['name']}: "
                    ).strip()
                    if value:
                        params["env"][env_var] = value
            else:
                value = input(
                    f"Please set the environment variable {env_var} for {srv['name']}: "
                ).strip()
                params["env"][env_var] = value
        # Add any extra env if present
        if "env_extra" in srv:
            params["env"].update(srv["env_extra"])
        if not exe_name or params["env"]:
            server_entry = {
                "name": srv["name"],
                "key": srv["key"],
                "params": params,
            }
            servers.append(server_entry)
    with open(actual_path, "w", encoding="utf-8") as f:
        json.dump({"servers": servers}, f, indent=2)
    print(f"Wrote {len(servers)} servers to {actual_path}")


write_mcp_json_from_servers(
    SCANNER_MCP_SERVERS, os.path.join(os.getcwd(), "scanner_mcp.json")
)
write_mcp_json_from_servers(
    PLANNER_MCP_SERVERS, os.path.join(os.getcwd(), "planner_mcp.json")
)
