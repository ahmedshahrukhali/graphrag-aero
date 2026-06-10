module.exports = {
  apps: [
    {
      name: "wake-proxy",
      script: "scripts/wake_proxy.py",
      interpreter: "python",
      cwd: "./"
    },
    {
      // cmd needs /c or it ignores the script and idles at an interactive prompt
      name: "localtunnel",
      script: "scripts\\run_localtunnel.bat",
      interpreter: "cmd",
      interpreter_args: "/c",
      cwd: "./"
    }
    // cloudflared: DISABLED. It opened a *second* public URL — a random
    // https://<random>.trycloudflare.com each run — competing with the fixed
    // localtunnel URL (graphrag-aero-cocko.loca.lt) the HF Space is configured
    // for. That random URL is the "different url" that popped up on `pm2 start`.
    // Re-enable only if you want a Cloudflare fallback tunnel.
    // ,{
    //   name: "cloudflared",
    //   script: "cloudflared.exe",
    //   args: "tunnel --url http://localhost:8080",
    //   interpreter: "none",
    //   cwd: "./"
    // }
  ]
};
