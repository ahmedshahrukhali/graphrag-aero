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
    },
    {
      name: "cloudflared",
      script: "cloudflared.exe",
      args: "tunnel --url http://localhost:8080",
      interpreter: "none",
      cwd: "./"
    }
  ]
};
