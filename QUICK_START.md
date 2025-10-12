# 🚀 IAAS Application - Quick Start

## **One-Command Setup & Start**

```bash
cd /Users/mostafa/newFianlLLMRepo/IAAS/client
npm start
```

That's it! This will automatically start all three servers:
- 🔵 **Frontend** (React) on `http://localhost:3000`
- 🟢 **Backend** (Node.js) on `http://localhost:5001`
- 🟣 **LLM Server** (Flask) on `http://localhost:5002`

## **Before First Run**

1. **Start MySQL/XAMPP** (if not already running)
2. **Run the setup checker** (optional but recommended):
   ```bash
   cd /Users/mostafa/newFianlLLMRepo/IAAS/client
   ./setup.sh
   ```

## **Login**

- **URL:** `http://localhost:3000`
- **Test Account:** `mostafa@gmail.com` (or any account in your database)

## **Troubleshooting**

- **Can't log in?** → Check if backend server is running on port 5001
- **PDF upload fails?** → Check if LLM server is running on port 5002  
- **Any errors?** → See detailed guide: `IAAS/client/TROUBLESHOOTING.md`

## **Stop Everything**

Press `Ctrl+C` in the terminal where you ran `npm start`

---

**📖 Full Documentation:** `IAAS/client/STARTUP.md`  
**🔧 Troubleshooting:** `IAAS/client/TROUBLESHOOTING.md`
