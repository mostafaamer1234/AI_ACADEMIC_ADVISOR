# 🚀 Application Startup Guide

## Quick Start

To start both the frontend and backend servers together, simply run:

```bash
cd /Users/mostafa/newFianlLLMRepo/IAAS/client
npm start
```

This single command will automatically start:
- **Frontend (React)** on `http://localhost:3000` (cyan label)
- **Backend (Node.js)** on `http://localhost:5001` (green label)
- **LLM Server (Flask)** on `http://localhost:5002` (magenta label)

## Individual Server Commands

If you need to run servers separately:

### Start only the frontend:
```bash
npm run start:frontend
```

### Start only the backend:
```bash
npm run start:backend
```

### Start only the LLM server:
```bash
npm run start:llm
```

## Prerequisites

Before running the application, ensure:

1. ✅ **MySQL/XAMPP is running** on port 3306
2. ✅ **Database `Advising` exists** with the `users` table
3. ✅ **Dependencies are installed** (`npm install` in `/IAAS/client`)

### 🚀 **Quick Setup (Recommended)**

Run the setup script to check everything automatically:
```bash
cd /Users/mostafa/newFianlLLMRepo/IAAS/client
./setup.sh
```

This will verify all prerequisites and guide you through any issues.

## Troubleshooting

### 🔧 **Need Help?**

For detailed troubleshooting, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

### Quick Fixes:
- **Login issues:** Check if backend server is running on port 5001
- **PDF upload errors:** Check if LLM server is running on port 5002
- **CORS errors:** Restart all servers with `npm start`
- **Python errors:** Use `python3` (already configured)

### Port Already in Use
```bash
# Kill all application processes
lsof -ti:3000,5001,5002 | xargs kill 2>/dev/null

# Then restart
npm start
```

## How It Works

The `npm start` command uses `concurrently` to run all three servers simultaneously. You'll see color-coded output:
- 🔵 **[FRONTEND]** - React development server output
- 🟢 **[BACKEND]** - Node.js server output  
- 🟣 **[LLM]** - Flask LLM server output

All servers will continue running until you press `Ctrl+C` to stop them.

