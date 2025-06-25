from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="MathServer", stateless=True)

@mcp.tool(description="A simple add tool")
def add_two(n: int) -> int:
  return n + 2