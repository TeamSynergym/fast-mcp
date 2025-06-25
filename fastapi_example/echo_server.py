from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="EchoServer", stateless=True)

@mcp.tool(description="A simple echo tool")
def echo(message: str) -> str:
  return f"Echo: {message}"


@mcp.tool(description="A tool that returns a secret key")
def secret_key() -> str:
  return "the secret key is 'Charllote Rezbach'"